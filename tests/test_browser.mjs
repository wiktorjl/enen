import { spawn, spawnSync } from "node:child_process";
import { readFile, mkdtemp, rm } from "node:fs/promises";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { dirname, extname, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const WEB_ROOT = resolve(ROOT, "webapp");
const STARTUP_DEADLINE_MS = 20000;
const TRAINING_DEADLINE_MS = 45000;
const EXPECTED_MODEL_BYTES = 28 + 4546 * Float64Array.BYTES_PER_ELEMENT;

const [canonicalSourceBytes, browserSourceBytes, stylesheetSource] = await Promise.all([
  readFile(resolve(ROOT, "src", "nn.c")),
  readFile(resolve(WEB_ROOT, "assets", "nn.c")),
  readFile(resolve(WEB_ROOT, "styles.css"), "utf8"),
]);
if (!canonicalSourceBytes.equals(browserSourceBytes)) {
  throw new Error("Browser nn.c asset differs byte-for-byte from src/nn.c");
}
const canonicalSource = canonicalSourceBytes.toString("utf8");

const mimeTypes = new Map([
  [".c", "text/plain; charset=utf-8"],
  [".css", "text/css; charset=utf-8"],
  [".html", "text/html; charset=utf-8"],
  [".js", "text/javascript; charset=utf-8"],
  [".tes", "application/octet-stream"],
  [".tra", "application/octet-stream"],
  [".wasm", "application/wasm"],
]);

const delay = (milliseconds) => new Promise((resolveDelay) => {
  setTimeout(resolveDelay, milliseconds);
});

function check(condition, message, details = undefined) {
  if (!condition) {
    const suffix = details === undefined ? "" : `: ${JSON.stringify(details)}`;
    throw new Error(`${message}${suffix}`);
  }
}

function maximumDelta(left, right) {
  check(left.length === right.length, "Cannot compare arrays with different lengths", {
    left: left.length,
    right: right.length,
  });
  let delta = 0;
  for (let index = 0; index < left.length; index += 1) {
    delta = Math.max(delta, Math.abs(left[index] - right[index]));
  }
  return delta;
}

for (const [label, pattern] of [
  ["gradient", /(?:linear|radial|conic)-gradient\s*\(/i],
  ["box shadow", /box-shadow\s*:/i],
  ["text shadow", /text-shadow\s*:/i],
  ["filter", /(?:backdrop-)?filter\s*:/i],
]) {
  check(!pattern.test(stylesheetSource), `Flat stylesheet contains a ${label}`);
}
const radiusValues = Array.from(
  stylesheetSource.matchAll(/border-radius\s*:\s*([^;}\n]+)/gi),
  (match) => match[1].trim(),
);
check(
  radiusValues.length > 0 && radiusValues.every((value) => /^0(?:px|rem|em|%)?$/.test(value)),
  "Flat stylesheet contains a nonzero border radius",
  radiusValues,
);

function findBrowser() {
  const candidates = [
    process.env.BROWSER,
    "chromium",
    "chromium-browser",
    "google-chrome",
    "google-chrome-stable",
  ].filter(Boolean);
  for (const candidate of candidates) {
    const result = spawnSync(candidate, ["--version"], { stdio: "ignore" });
    if (result.status === 0) return candidate;
  }
  throw new Error(
    "No Chromium-based browser found; set BROWSER to a Chrome or Chromium executable",
  );
}

async function startServer() {
  const server = createServer(async (request, response) => {
    try {
      const url = new URL(request.url, "http://127.0.0.1");
      const pathname = decodeURIComponent(url.pathname === "/" ? "/index.html" : url.pathname);
      const filename = resolve(WEB_ROOT, `.${pathname}`);
      if (filename !== WEB_ROOT && !filename.startsWith(`${WEB_ROOT}${sep}`)) {
        response.writeHead(403).end("Forbidden");
        return;
      }
      const body = await readFile(filename);
      response.writeHead(200, {
        "Cache-Control": "no-store",
        "Content-Type": mimeTypes.get(extname(filename)) || "application/octet-stream",
      });
      response.end(body);
    } catch {
      response.writeHead(404).end("Not found");
    }
  });
  await new Promise((resolveListen, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolveListen);
  });
  return server;
}

class DevToolsPipe {
  constructor(browser) {
    this.browser = browser;
    this.input = browser.stdio[3];
    this.output = browser.stdio[4];
    this.nextId = 1;
    this.pending = new Map();
    this.buffer = "";
    this.output.setEncoding("utf8");
    this.output.on("data", (chunk) => this.receive(chunk));
    browser.once("exit", (code) => {
      for (const { reject } of this.pending.values()) {
        reject(new Error(`Browser exited unexpectedly with status ${code}`));
      }
      this.pending.clear();
    });
  }

  receive(chunk) {
    this.buffer += chunk;
    const messages = this.buffer.split("\0");
    this.buffer = messages.pop();
    for (const message of messages) {
      if (!message) continue;
      const parsed = JSON.parse(message);
      const pending = this.pending.get(parsed.id);
      if (!pending) continue;
      this.pending.delete(parsed.id);
      if (parsed.error) pending.reject(new Error(parsed.error.message));
      else pending.resolve(parsed.result);
    }
  }

  call(method, params = {}, sessionId = undefined) {
    const id = this.nextId;
    this.nextId += 1;
    return new Promise((resolveCall, reject) => {
      this.pending.set(id, { resolve: resolveCall, reject });
      this.input.write(`${JSON.stringify({ id, method, params, sessionId })}\0`);
    });
  }
}

async function evaluate(devtools, sessionId, expression) {
  const result = await devtools.call(
    "Runtime.evaluate",
    { expression, returnByValue: true, awaitPromise: true },
    sessionId,
  );
  if (result.exceptionDetails) {
    const description = result.exceptionDetails.exception?.description;
    throw new Error(description || result.exceptionDetails.text || "Browser evaluation failed");
  }
  return result.result.value;
}

async function waitFor(devtools, sessionId, expression, accept, deadline, label) {
  let lastValue;
  const expiresAt = Date.now() + deadline;
  while (Date.now() < expiresAt) {
    try {
      lastValue = await evaluate(devtools, sessionId, expression);
      if (accept(lastValue)) return lastValue;
    } catch (error) {
      // A reload briefly destroys the page's execution context. Continue until
      // the new document is ready, while retaining the last error for timeout diagnostics.
      lastValue = { evaluationError: error.message };
    }
    await delay(50);
  }
  throw new Error(`${label} timed out; last browser state: ${JSON.stringify(lastValue)}`);
}

async function dispatchKey(devtools, sessionId, key, code, keyCode, modifiers = 0) {
  await devtools.call("Input.dispatchKeyEvent", {
    type: "rawKeyDown",
    key,
    code,
    windowsVirtualKeyCode: keyCode,
    nativeVirtualKeyCode: keyCode,
    modifiers,
  }, sessionId);
  await devtools.call("Input.dispatchKeyEvent", {
    type: "keyUp",
    key,
    code,
    windowsVirtualKeyCode: keyCode,
    nativeVirtualKeyCode: keyCode,
    modifiers,
  }, sessionId);
}

async function dispatchPointerDrag(devtools, sessionId, start, end, steps = 20) {
  await devtools.call("Input.dispatchMouseEvent", {
    type: "mouseMoved",
    x: start.x,
    y: start.y,
    pointerType: "mouse",
  }, sessionId);
  await devtools.call("Input.dispatchMouseEvent", {
    type: "mousePressed",
    x: start.x,
    y: start.y,
    button: "left",
    buttons: 1,
    clickCount: 1,
    pointerType: "mouse",
  }, sessionId);
  for (let step = 1; step <= steps; step += 1) {
    const progress = step / steps;
    await devtools.call("Input.dispatchMouseEvent", {
      type: "mouseMoved",
      x: start.x + (end.x - start.x) * progress,
      y: start.y + (end.y - start.y) * progress,
      button: "left",
      buttons: 1,
      pointerType: "mouse",
    }, sessionId);
  }
  await devtools.call("Input.dispatchMouseEvent", {
    type: "mouseReleased",
    x: end.x,
    y: end.y,
    button: "left",
    buttons: 0,
    clickCount: 1,
    pointerType: "mouse",
  }, sessionId);
}

async function stopBrowser(browser) {
  if (browser.exitCode !== null) return;
  browser.kill("SIGTERM");
  await Promise.race([
    new Promise((resolveExit) => browser.once("exit", resolveExit)),
    delay(1000),
  ]);
  if (browser.exitCode === null) browser.kill("SIGKILL");
}

const modelAuditExpression = `(async () => {
  const record = await readStoredModel();
  if (!record) return { missing: true };
  const path = "/browser-acceptance.model";
  try { state.wasm.FS.unlink(path); } catch {}
  const saved = state.wasm.ccall(
    "web_save_model", "number", ["string"], [path]
  );
  const serialized = saved === 0 ? state.wasm.FS.readFile(path) : new Uint8Array();
  const stored = record.model instanceof ArrayBuffer
    ? new Uint8Array(record.model)
    : new Uint8Array();
  let exact = serialized.length === stored.length;
  let firstDifference = -1;
  let storedHash = 2166136261;
  let serializedHash = 2166136261;
  const length = Math.max(stored.length, serialized.length);
  for (let index = 0; index < length; index += 1) {
    if (exact && stored[index] !== serialized[index]) {
      exact = false;
      firstDifference = index;
    }
    if (index < stored.length) {
      storedHash = Math.imul(storedHash ^ stored[index], 16777619);
    }
    if (index < serialized.length) {
      serializedHash = Math.imul(serializedHash ^ serialized[index], 16777619);
    }
  }
  try { state.wasm.FS.unlink(path); } catch {}
  return {
    missing: false,
    version: record.version,
    revision: record.revision,
    savedAt: record.savedAt,
    modelIsArrayBuffer: record.model instanceof ArrayBuffer,
    storedLength: stored.length,
    serializedLength: serialized.length,
    saved,
    exact,
    firstDifference,
    storedHash: storedHash >>> 0,
    serializedHash: serializedHash >>> 0,
    config: record.config,
    metrics: record.metrics,
  };
})()`;

const fixturePredictionsExpression = `(() => {
  const indices = [0, 1, 50, 500, 1796];
  return indices.map((index) => {
    const label = state.wasm._web_copy_test_sample(index);
    const prediction = state.wasm._web_predict();
    const probabilities = Array.from(
      { length: 10 },
      (_, digit) => state.wasm._web_probability(digit),
    );
    return { index, label, prediction, probabilities };
  });
})()`;

const readyExpression = `(() => ({
  runtime: document.querySelector("#runtime-status")?.textContent || "missing",
  status: document.querySelector("#training-status")?.textContent || "missing",
  disabled: document.querySelector("#train-button")?.disabled ?? true,
  resetDisabled: document.querySelector("#new-model-button")?.disabled ?? true,
  stylesheets: document.styleSheets.length,
  reducedMotion: typeof state === "undefined" ? null : state.reducedMotion,
  origin: location.origin
}))()`;

const server = await startServer();
const profile = await mkdtemp(resolve(tmpdir(), "enen-browser-check-"));
const address = server.address();
const pageUrl = `http://127.0.0.1:${address.port}/`;
const expectedOrigin = new URL(pageUrl).origin;
const browser = spawn(findBrowser(), [
  "--headless=new",
  "--no-sandbox",
  "--disable-gpu",
  "--remote-debugging-pipe",
  `--user-data-dir=${profile}`,
  "about:blank",
], {
  stdio: ["ignore", "ignore", "pipe", "pipe", "pipe"],
});

try {
  const devtools = new DevToolsPipe(browser);
  let page;
  for (let attempt = 0; attempt < 100 && !page; attempt += 1) {
    const { targetInfos } = await devtools.call("Target.getTargets");
    page = targetInfos.find((target) => target.type === "page");
    if (!page) await delay(25);
  }
  if (!page) throw new Error("Browser page target did not appear");
  const { sessionId } = await devtools.call("Target.attachToTarget", {
    targetId: page.targetId,
    flatten: true,
  });
  await devtools.call("Runtime.enable", {}, sessionId);
  await devtools.call("Page.enable", {}, sessionId);
  await devtools.call("Emulation.setDeviceMetricsOverride", {
    width: 1440,
    height: 900,
    deviceScaleFactor: 1,
    mobile: false,
  }, sessionId);
  await devtools.call("Emulation.setEmulatedMedia", {
    features: [{ name: "prefers-reduced-motion", value: "no-preference" }],
  }, sessionId);
  const navigation = await devtools.call("Page.navigate", { url: pageUrl }, sessionId);
  if (navigation.errorText) {
    throw new Error(`Browser navigation failed: ${navigation.errorText}`);
  }

  const ready = await waitFor(
    devtools,
    sessionId,
    readyExpression,
    (value) => value.runtime === "C / Wasm ready" || value.runtime === "Load failed",
    STARTUP_DEADLINE_MS,
    "Clean browser startup",
  );
  check(
    ready.runtime === "C / Wasm ready" && !ready.disabled && !ready.resetDisabled &&
      ready.stylesheets >= 1 && ready.reducedMotion === false &&
      ready.origin === expectedOrigin,
    "Browser did not become ready on the expected origin",
    ready,
  );

  const initialConfiguration = await evaluate(
    devtools,
    sessionId,
    `(async () => ({
      architectures: Array.from(
        document.querySelector("#architecture").options,
        (option) => option.value,
      ),
      architecture: document.querySelector("#architecture").value,
      epochMin: document.querySelector("#epochs").min,
      epochMax: document.querySelector("#epochs").max,
      epochStep: document.querySelector("#epochs").step,
      epochs: document.querySelector("#epochs").value,
      learningRates: Array.from(
        document.querySelector("#learning-rate").options,
        (option) => option.value,
      ),
      learningRate: document.querySelector("#learning-rate").value,
      locked: document.querySelector("#recognition-panel").getAttribute("aria-disabled"),
      drawingDisabled: document.querySelector("#drawing-canvas").getAttribute("aria-disabled"),
      drawToolDisabled: document.querySelector("#draw-tool-button").disabled,
      eraserToolDisabled: document.querySelector("#eraser-tool-button").disabled,
      drawPressed: document.querySelector("#draw-tool-button").getAttribute("aria-pressed"),
      eraserPressed: document.querySelector("#eraser-tool-button").getAttribute("aria-pressed"),
      drawingTool: state.drawingTool,
      canvasTool: document.querySelector("#drawing-canvas").dataset.tool,
      clearDisabled: document.querySelector("#clear-button").disabled,
      exampleDisabled: document.querySelector("#example-button").disabled,
      storedModelLoaded: state.storedModelLoaded,
      storedRecordAbsent: (await readStoredModel()) === undefined,
      storageStatus: document.querySelector("#storage-status").textContent,
      shape: modelShape(),
      epochsTrained: state.wasm._web_epochs_trained(),
    }))()`,
  );
  check(
    initialConfiguration.architectures.join("|") === "24,16|32,16|48,24|128,64" &&
      initialConfiguration.architecture === "128,64" &&
      initialConfiguration.epochMin === "10" &&
      initialConfiguration.epochMax === "100" &&
      initialConfiguration.epochStep === "1" &&
      initialConfiguration.epochs === "25" &&
      initialConfiguration.learningRates.join("|") === "0.03|0.05|0.08" &&
      initialConfiguration.learningRate === "0.05" &&
      initialConfiguration.locked === "true" &&
      initialConfiguration.drawingDisabled === "true" &&
      initialConfiguration.drawToolDisabled && initialConfiguration.eraserToolDisabled &&
      initialConfiguration.drawPressed === "true" &&
      initialConfiguration.eraserPressed === "false" &&
      initialConfiguration.drawingTool === "draw" &&
      initialConfiguration.canvasTool === "draw" &&
      initialConfiguration.clearDisabled && initialConfiguration.exampleDisabled &&
      !initialConfiguration.storedModelLoaded && initialConfiguration.storedRecordAbsent &&
      initialConfiguration.storageStatus.includes("No saved model") &&
      initialConfiguration.shape.join(",") === "64,128,64,10" &&
      initialConfiguration.epochsTrained === 0,
    "Fresh profile did not start from the clean reference-model state",
    initialConfiguration,
  );

  const tabStateExpression = `(() => {
    const entries = [
      ["train", "train-tab", "train-panel"],
      ["recognize", "recognize-tab", "recognition-panel"],
    ];
    return {
      tabCount: document.querySelectorAll('[role="tab"]').length,
      activeElement: document.activeElement?.id || "",
      entries: entries.map(([name, tabId, panelId]) => {
        const tab = document.getElementById(tabId);
        const panel = document.getElementById(panelId);
        return {
          name,
          selected: tab.getAttribute("aria-selected"),
          tabIndex: tab.tabIndex,
          controls: tab.getAttribute("aria-controls"),
          panelId,
          hidden: panel.hidden,
        };
      }),
    };
  })()`;
  function checkTabState(snapshot, selectedName, expectFocus = false) {
    const selected = snapshot.entries.find((entry) => entry.name === selectedName);
    check(selected && snapshot.tabCount === 2, `Missing final two-tab state for ${selectedName}`, snapshot);
    check(
      snapshot.entries.filter((entry) => entry.selected === "true").length === 1 &&
        snapshot.entries.filter((entry) => !entry.hidden).length === 1 &&
        selected.selected === "true" && selected.tabIndex === 0 && !selected.hidden &&
        selected.controls === selected.panelId &&
        snapshot.entries
          .filter((entry) => entry !== selected)
          .every((entry) => entry.selected === "false" && entry.tabIndex === -1 && entry.hidden),
      `Tabs did not select only ${selectedName}`,
      snapshot,
    );
    if (expectFocus) {
      check(
        snapshot.activeElement === `${selectedName}-tab`,
        `${selectedName} tab did not receive keyboard focus`,
        snapshot,
      );
    }
  }

  const networkStateExpression = `(() => {
    const canvases = document.querySelectorAll("#network-canvas");
    const canvas = canvases[0];
    const panel = document.querySelector(".network-panel");
    const bounds = canvas?.getBoundingClientRect();
    const style = canvas ? getComputedStyle(canvas) : null;
    return {
      count: canvases.length,
      oldTrainingCanvas: Boolean(document.querySelector("#training-network-canvas")),
      oldRecognitionCanvas: Boolean(document.querySelector("#recognition-network-canvas")),
      visible: Boolean(canvas && bounds.width > 0 && bounds.height > 0 &&
        style.display !== "none" && style.visibility !== "hidden" && !panel.hidden),
      width: bounds?.width || 0,
      height: bounds?.height || 0,
      rendererPresent: Boolean(networkRenderer),
      rendererSame: window.__enenUnifiedIdentity
        ? window.__enenUnifiedIdentity.renderer === networkRenderer &&
          window.__enenUnifiedIdentity.canvas === canvas
        : true,
      renderCount: Number(canvas?.dataset.renderCount || 0),
      version: Number(canvas?.dataset.activationVersion || 0),
      shape: networkRenderer?.snapshot?.layerSizes || [],
    };
  })()`;

  checkTabState(await evaluate(devtools, sessionId, tabStateExpression), "train");
  const initialNetwork = await evaluate(
    devtools,
    sessionId,
    `(() => {
      window.__enenUnifiedIdentity = {
        canvas: document.querySelector("#network-canvas"),
        renderer: networkRenderer,
      };
      return ${networkStateExpression};
    })()`,
  );
  check(
    initialNetwork.count === 1 && !initialNetwork.oldTrainingCanvas &&
      !initialNetwork.oldRecognitionCanvas && initialNetwork.visible &&
      initialNetwork.rendererPresent && initialNetwork.renderCount > 0 &&
      initialNetwork.shape.join(",") === "64,128,64,10",
    "The clean app did not expose one visible unified network monitor",
    initialNetwork,
  );

  const computedFlatness = await evaluate(
    devtools,
    sessionId,
    `(() => {
      const selectors = [
        "body", ".lab-header", ".control-panel", ".network-panel",
        ".options-form fieldset", ".drawing-card", ".prediction-card",
        "#source-button", ".primary-button", "progress", "#network-canvas",
        ".source-window", ".source-view"
      ];
      return selectors.map((selector) => {
        const element = document.querySelector(selector);
        const style = getComputedStyle(element);
        return {
          selector,
          backgroundImage: style.backgroundImage,
          boxShadow: style.boxShadow,
          textShadow: style.textShadow,
          filter: style.filter,
          backdropFilter: style.backdropFilter || "none",
          borderRadius: style.borderRadius,
        };
      });
    })()`,
  );
  check(
    computedFlatness.every((style) =>
      style.backgroundImage === "none" && style.boxShadow === "none" &&
      style.textShadow === "none" && style.filter === "none" &&
      style.backdropFilter === "none" && style.borderRadius === "0px"),
    "Computed interface styles violate the flat visual contract",
    computedFlatness,
  );

  await evaluate(
    devtools,
    sessionId,
    `(() => {
      const button = document.querySelector("#source-button");
      button.focus();
      button.click();
    })()`,
  );
  const sourceView = await waitFor(
    devtools,
    sessionId,
    `(() => ({
      open: document.querySelector("#source-dialog").open,
      loaded: state.sourceLoaded,
      loading: state.sourceLoading,
      focusInside: document.querySelector("#source-dialog").contains(document.activeElement),
      activeElement: document.activeElement?.id || document.activeElement?.className || "",
      status: document.querySelector("#source-status").textContent,
      source: document.querySelector("#source-code").textContent
    }))()`,
    (value) => value.open && value.loaded,
    STARTUP_DEADLINE_MS,
    "Canonical source dialog",
  );
  check(
    sourceView.open && sourceView.focusInside && sourceView.status.includes("shown in full") &&
      sourceView.source === canonicalSource,
    "Source dialog did not focus and render canonical src/nn.c in full",
    { ...sourceView, source: `${sourceView.source.length} bytes` },
  );
  const rawSource = await evaluate(
    devtools,
    sessionId,
    `(async () => {
      const response = await fetch("assets/nn.c", { cache: "no-store" });
      const link = document.querySelector('#source-dialog a[href="assets/nn.c"]');
      return {
        ok: response.ok,
        type: response.headers.get("content-type"),
        source: await response.text(),
        href: link.href,
        target: link.target,
        rel: link.rel,
      };
    })()`,
  );
  check(
    rawSource.ok && rawSource.type.startsWith("text/plain") &&
      rawSource.source === canonicalSource && rawSource.href.endsWith("/assets/nn.c") &&
      rawSource.target === "_blank" && rawSource.rel.includes("noreferrer"),
    "Raw source asset, MIME type, or dialog link is incorrect",
    { ...rawSource, source: `${rawSource.source.length} bytes` },
  );
  const dialogFocusCycle = [];
  for (let press = 0; press < 5; press += 1) {
    await dispatchKey(devtools, sessionId, "Tab", "Tab", 9);
    dialogFocusCycle.push(await evaluate(
      devtools,
      sessionId,
      `(() => ({
        inside: document.querySelector("#source-dialog").contains(document.activeElement),
        id: document.activeElement?.id || "",
        tag: document.activeElement?.tagName || "",
        modal: document.querySelector("#source-dialog").matches(":modal"),
        dialogOpen: document.querySelector("#source-dialog").open,
      }))()`,
    ));
  }
  check(
    dialogFocusCycle.every((entry) =>
      entry.modal && entry.dialogOpen && (entry.inside || entry.tag === "BODY")) &&
      dialogFocusCycle.every((entry, index) =>
        entry.tag !== "BODY" || dialogFocusCycle[index + 1]?.inside),
    "Tab reached an interactive background element outside the modal source dialog",
    dialogFocusCycle,
  );
  await dispatchKey(devtools, sessionId, "Tab", "Tab", 9, 8);
  check(
    await evaluate(
      devtools,
      sessionId,
      "document.querySelector('#source-dialog').contains(document.activeElement)",
    ),
    "Shift+Tab moved focus outside the modal source dialog",
  );
  await dispatchKey(devtools, sessionId, "Escape", "Escape", 27);
  const sourceClosed = await waitFor(
    devtools,
    sessionId,
    `(() => ({
      open: document.querySelector("#source-dialog").open,
      activeElement: document.activeElement?.id || ""
    }))()`,
    (value) => !value.open,
    3000,
    "Source dialog Escape close",
  );
  check(
    sourceClosed.activeElement === "source-button",
    "Escape did not return focus to the source-dialog invoker",
    sourceClosed,
  );

  await evaluate(devtools, sessionId, "document.querySelector('#train-tab').focus()");
  await dispatchKey(devtools, sessionId, "ArrowRight", "ArrowRight", 39);
  checkTabState(await evaluate(devtools, sessionId, tabStateExpression), "recognize", true);
  const lockedRecognition = await evaluate(
    devtools,
    sessionId,
    `(() => ({
      panelDisabled: document.querySelector("#recognition-panel").getAttribute("aria-disabled"),
      canvasDisabled: document.querySelector("#drawing-canvas").getAttribute("aria-disabled"),
      drawToolDisabled: document.querySelector("#draw-tool-button").disabled,
      eraserToolDisabled: document.querySelector("#eraser-tool-button").disabled,
      drawPressed: document.querySelector("#draw-tool-button").getAttribute("aria-pressed"),
      eraserPressed: document.querySelector("#eraser-tool-button").getAttribute("aria-pressed"),
      drawingTool: state.drawingTool,
      canvasTool: document.querySelector("#drawing-canvas").dataset.tool,
      clearDisabled: document.querySelector("#clear-button").disabled,
      exampleDisabled: document.querySelector("#example-button").disabled,
      network: ${networkStateExpression}
    }))()`,
  );
  check(
    lockedRecognition.panelDisabled === "true" &&
      lockedRecognition.canvasDisabled === "true" &&
      lockedRecognition.drawToolDisabled && lockedRecognition.eraserToolDisabled &&
      lockedRecognition.drawPressed === "true" && lockedRecognition.eraserPressed === "false" &&
      lockedRecognition.drawingTool === "draw" && lockedRecognition.canvasTool === "draw" &&
      lockedRecognition.clearDisabled && lockedRecognition.exampleDisabled &&
      lockedRecognition.network.visible && lockedRecognition.network.rendererSame,
    "Locked Recognize mode exposed controls or hid/replaced the unified network",
    lockedRecognition,
  );
  await dispatchKey(devtools, sessionId, "Home", "Home", 36);
  checkTabState(await evaluate(devtools, sessionId, tabStateExpression), "train", true);
  await dispatchKey(devtools, sessionId, "End", "End", 35);
  checkTabState(await evaluate(devtools, sessionId, tabStateExpression), "recognize", true);
  await dispatchKey(devtools, sessionId, "ArrowLeft", "ArrowLeft", 37);
  checkTabState(await evaluate(devtools, sessionId, tabStateExpression), "train", true);

  const configured = await evaluate(
    devtools,
    sessionId,
    `(() => {
      const architecture = document.querySelector("#architecture");
      const epochs = document.querySelector("#epochs");
      const learningRate = document.querySelector("#learning-rate");
      architecture.value = "48,24";
      architecture.dispatchEvent(new Event("change", { bubbles: true }));
      epochs.value = "25";
      epochs.dispatchEvent(new Event("input", { bubbles: true }));
      learningRate.value = "0.05";
      learningRate.dispatchEvent(new Event("change", { bubbles: true }));
      document.querySelector("#new-model-button").click();
      return {
        architecture: architecture.value,
        epochs: epochs.value,
        epochOutput: document.querySelector("#epochs-value").value,
        learningRate: learningRate.value,
        architectureNote: document.querySelector("#architecture-note").textContent,
        layerSummary: document.querySelector("#layer-summary").textContent,
        networkShape: document.querySelector("#network-shape").textContent,
        shape: modelShape(),
        trainedEpochs: state.wasm._web_epochs_trained(),
        locked: document.querySelector("#recognition-panel").classList.contains("is-disabled"),
        metricValues: [
          document.querySelector("#test-accuracy").textContent,
          document.querySelector("#test-loss").textContent,
          document.querySelector("#drawing-accuracy").textContent,
        ],
        currentConfig: state.currentConfig,
        network: ${networkStateExpression},
      };
    })()`,
  );
  check(
    configured.architecture === "48,24" && configured.epochs === "25" &&
      configured.epochOutput === "25" && configured.learningRate === "0.05" &&
      configured.architectureNote.replace(/\D/g, "") === "4546" &&
      configured.layerSummary === "64 → 48 → 24 → 10" &&
      configured.networkShape === "64 → 48 → 24 → 10" &&
      configured.shape.join(",") === "64,48,24,10" &&
      configured.trainedEpochs === 0 && configured.locked &&
      configured.metricValues.every((value) => value === "—") &&
      configured.currentConfig.firstHidden === 48 &&
      configured.currentConfig.secondHidden === 24 &&
      configured.currentConfig.epochs === 25 &&
      configured.currentConfig.learningRate === 0.05 &&
      configured.network.visible && configured.network.rendererSame &&
      configured.network.shape.join(",") === "64,48,24,10",
    "Nondefault model was not configured through the final UI",
    configured,
  );

  await evaluate(
    devtools,
    sessionId,
    `(() => {
      const wasm = state.wasm;
      const audit = window.__enenBrowserAudit = {
        batches: [],
        evaluationCalls: 0,
        started: 0,
      };
      if (wasm._web_train_batch(0, 0.05) !== -1) {
        throw new Error("C batch API accepted a zero-sized batch");
      }
      const trainBatch = wasm._web_train_batch;
      wasm._web_train_batch = (maxSamples, learningRate) => {
        const processed = trainBatch(maxSamples, learningRate);
        audit.batches.push({
          at: performance.now(),
          maxSamples,
          learningRate,
          processed,
          version: wasm._web_activation_version(),
          label: wasm._web_last_training_label(),
          epochs: wasm._web_epochs_trained(),
          position: wasm._web_epoch_position(),
        });
        return processed;
      };
      const evaluateModel = wasm._web_evaluate;
      wasm._web_evaluate = () => {
        audit.evaluationCalls += 1;
        return evaluateModel();
      };
    })()`,
  );
  const trainingStarted = await evaluate(
    devtools,
    sessionId,
    `(() => {
      window.__enenBrowserAudit.started = performance.now();
      document.querySelector("#train-button").click();
      return window.__enenBrowserAudit.started;
    })()`,
  );
  const stage = await waitFor(
    devtools,
    sessionId,
    `(() => {
      const canvas = document.querySelector("#network-canvas");
      const bounds = canvas.getBoundingClientRect();
      return {
        training: state.training,
        setupHidden: document.querySelector("#setup-view").hidden,
        stageHidden: document.querySelector("#training-stage").hidden,
        architectureDisabled: document.querySelector("#architecture").disabled,
        epochsDisabled: document.querySelector("#epochs").disabled,
        learningRateDisabled: document.querySelector("#learning-rate").disabled,
        buttonDisabled: document.querySelector("#train-button").disabled,
        networkVisible: bounds.width > 0 && bounds.height > 0 &&
          getComputedStyle(canvas).display !== "none",
        networkSame: window.__enenUnifiedIdentity.renderer === networkRenderer &&
          window.__enenUnifiedIdentity.canvas === canvas,
      };
    })()`,
    (value) => value.training && value.setupHidden && !value.stageHidden,
    STARTUP_DEADLINE_MS,
    "Animated training stage",
  );
  check(
    stage.architectureDisabled && stage.epochsDisabled &&
      stage.learningRateDisabled && stage.buttonDisabled &&
      stage.networkVisible && stage.networkSame,
    "Training did not lock configuration while retaining the unified network",
    stage,
  );

  const observedFrames = [];
  const observedVersions = new Set();
  const animationDeadline = Date.now() + 12000;
  let lastAnimationMarker;
  while (observedFrames.length < 2 && Date.now() < animationDeadline) {
    lastAnimationMarker = await evaluate(
      devtools,
      sessionId,
      `(() => {
        const canvas = document.querySelector("#network-canvas");
        const version = Number(canvas.dataset.activationVersion || 0);
        return {
          runtime: document.querySelector("#runtime-status").textContent,
          training: state.training,
          renderCount: Number(canvas.dataset.renderCount || 0),
          version,
          backedByBatch: window.__enenBrowserAudit.batches.some(
            (batch) => batch.version === version && batch.maxSamples === 128
          ),
        };
      })()`,
    );
    if (lastAnimationMarker.runtime === "Training failed") break;
    if (lastAnimationMarker.version > 0 && lastAnimationMarker.backedByBatch &&
        !observedVersions.has(lastAnimationMarker.version)) {
      observedVersions.add(lastAnimationMarker.version);
      await delay(125);
      const frame = await evaluate(
        devtools,
        sessionId,
        `(() => {
          const snapshot = networkRenderer?.snapshot;
          const canvas = document.querySelector("#network-canvas");
          const context = canvas.getContext("2d");
          const rgba = context.getImageData(0, 0, canvas.width, canvas.height).data;
          let canvasHash = 2166136261;
          for (let index = 0; index < rgba.length; index += 16) {
            canvasHash = Math.imul(canvasHash ^ rgba[index], 16777619);
            canvasHash = Math.imul(canvasHash ^ rgba[index + 1], 16777619);
            canvasHash = Math.imul(canvasHash ^ rgba[index + 2], 16777619);
          }
          let activationHash = 2166136261;
          for (const activation of snapshot.activations) {
            activationHash = Math.imul(
              activationHash ^ Math.round(activation * 1000000000),
              16777619,
            );
          }
          const matchingBatch = window.__enenBrowserAudit.batches.find(
            (batch) => batch.version === snapshot.version,
          );
          const bounds = canvas.getBoundingClientRect();
          return {
            at: performance.now(),
            started: window.__enenBrowserAudit.started,
            training: state.training,
            renderCount: Number(canvas.dataset.renderCount || 0),
            version: Number(canvas.dataset.activationVersion || 0),
            snapshotVersion: snapshot.version,
            wasmVersion: state.wasm._web_activation_version(),
            shape: snapshot.layerSizes,
            activationCount: snapshot.activations.length,
            activationsFinite: snapshot.activations.every(Number.isFinite),
            activationMinimum: Math.min(...snapshot.activations),
            activationMaximum: Math.max(...snapshot.activations),
            outputSum: snapshot.activations.slice(-10).reduce(
              (sum, activation) => sum + activation,
              0,
            ),
            label: snapshot.label,
            labelCopy: document.querySelector("#network-sample-label").textContent,
            matchingBatch: matchingBatch || null,
            activationHash: activationHash >>> 0,
            canvasHash: canvasHash >>> 0,
            progress: document.querySelector("#training-progress").value,
            networkVisible: bounds.width > 0 && bounds.height > 0,
            networkSame: window.__enenUnifiedIdentity.renderer === networkRenderer &&
              window.__enenUnifiedIdentity.canvas === canvas,
          };
        })()`,
      );
      observedFrames.push(frame);
    }
    await delay(50);
  }
  check(
    observedFrames.length >= 2,
    "Did not observe two genuine C activation/network frames",
    { frames: observedFrames, marker: lastAnimationMarker },
  );
  for (const frame of observedFrames) {
    check(
      frame.training && frame.version > 0 &&
        frame.snapshotVersion === frame.version && frame.wasmVersion >= frame.version &&
        frame.shape.join(",") === "64,48,24,10" && frame.activationCount === 146 &&
        frame.activationsFinite && frame.activationMinimum >= 0 &&
        frame.activationMaximum <= 1 && Math.abs(frame.outputSum - 1) < 1e-9 &&
        frame.matchingBatch?.maxSamples === 128 && frame.matchingBatch.processed > 0 &&
        frame.matchingBatch.label === frame.label && frame.labelCopy === String(frame.label) &&
        frame.progress > 0 && frame.networkVisible && frame.networkSame,
      "Rendered animation frame was not backed by a valid C training batch",
      frame,
    );
  }
  const firstFrameDelay = observedFrames[0].at - trainingStarted;
  const frameSpacing = observedFrames[1].at - observedFrames[0].at;
  check(
    firstFrameDelay >= 300 && firstFrameDelay <= 1500 &&
      frameSpacing >= 300 && frameSpacing <= 1200,
    "C activation frames did not follow the half-second cadence",
    { firstFrameDelay, frameSpacing },
  );
  check(
    observedFrames[1].version > observedFrames[0].version &&
      observedFrames[1].renderCount > observedFrames[0].renderCount &&
      observedFrames[1].activationHash !== observedFrames[0].activationHash &&
      observedFrames[1].canvasHash !== observedFrames[0].canvasHash,
    "Successive training frames did not show changing C activations and network pixels",
    observedFrames,
  );

  const skipped = await evaluate(
    devtools,
    sessionId,
    `(() => {
      const button = document.querySelector("#skip-animation-button");
      const wasAvailable = state.training && !button.hidden && !button.disabled;
      button.click();
      return {
        wasAvailable,
        training: state.training,
        skipAnimation: state.skipAnimation,
        disabled: button.disabled,
      };
    })()`,
  );
  check(
    skipped.wasAvailable && skipped.training && skipped.skipAnimation && skipped.disabled,
    "Finish without animation did not switch live training to the fast path",
    skipped,
  );

  const trained = await waitFor(
    devtools,
    sessionId,
    `(() => ({
      runtime: document.querySelector("#runtime-status").textContent,
      test: document.querySelector("#test-accuracy").textContent,
      loss: document.querySelector("#test-loss").textContent,
      drawing: document.querySelector("#drawing-accuracy").textContent,
      storageStatus: document.querySelector("#storage-status").textContent,
      storedModelLoaded: state.storedModelLoaded,
      locked: document.querySelector("#recognition-panel").classList.contains("is-disabled"),
      ariaDisabled: document.querySelector("#recognition-panel").getAttribute("aria-disabled"),
      epochs: state.wasm._web_epochs_trained(),
      shape: modelShape(),
      currentConfig: state.currentConfig,
      completionHidden: document.querySelector("#completion-actions").hidden,
      recognizeHidden: document.querySelector("#go-recognize-button").hidden,
      stageHidden: document.querySelector("#training-stage").hidden,
      setupHidden: document.querySelector("#setup-view").hidden,
      evaluationCalls: window.__enenBrowserAudit.evaluationCalls,
      smallBatches: window.__enenBrowserAudit.batches.filter(
        (batch) => batch.maxSamples === 128
      ).length,
      fastBatches: window.__enenBrowserAudit.batches.filter(
        (batch) => batch.maxSamples === 100000
      ).length
    }))()`,
    (value) => value.runtime === "Model trained" || value.runtime === "Training failed",
    TRAINING_DEADLINE_MS,
    "Browser training",
  );
  check(
    trained.runtime === "Model trained" && !trained.locked &&
      trained.ariaDisabled === "false" && trained.epochs === 25 &&
      trained.shape.join(",") === "64,48,24,10" &&
      trained.currentConfig.firstHidden === 48 &&
      trained.currentConfig.secondHidden === 24 &&
      trained.currentConfig.epochs === 25 &&
      trained.currentConfig.learningRate === 0.05 &&
      parseFloat(trained.test) >= 94 && parseFloat(trained.drawing) >= 80 &&
      Number.isFinite(Number(trained.loss)) &&
      !trained.completionHidden && !trained.recognizeHidden &&
      !trained.stageHidden && trained.setupHidden &&
      trained.evaluationCalls === 1 && trained.smallBatches > 0 && trained.fastBatches > 0 &&
      trained.storedModelLoaded && trained.storageStatus.includes("Saved in this browser profile"),
    "Browser nondefault training failed its completion/persistence contract",
    trained,
  );

  const storedModel = await evaluate(devtools, sessionId, modelAuditExpression);
  check(
    !storedModel.missing && storedModel.version === 2 &&
      typeof storedModel.revision === "string" && storedModel.revision.length >= 8 &&
      Number.isFinite(storedModel.savedAt) && storedModel.savedAt > 0 &&
      storedModel.modelIsArrayBuffer && storedModel.saved === 0 && storedModel.exact &&
      storedModel.firstDifference === -1 &&
      storedModel.storedLength === EXPECTED_MODEL_BYTES &&
      storedModel.serializedLength === EXPECTED_MODEL_BYTES &&
      storedModel.storedHash === storedModel.serializedHash &&
      storedModel.config.firstHidden === 48 && storedModel.config.secondHidden === 24 &&
      storedModel.config.epochs === 25 && storedModel.config.learningRate === 0.05 &&
      Number.isInteger(storedModel.config.seed) &&
      storedModel.metrics.epochs === 25 &&
      storedModel.metrics.testAccuracy >= 0.94 &&
      storedModel.metrics.drawingAccuracy >= 0.8 &&
      storedModel.metrics.minimumClassAccuracy >= 0.65 &&
      Number.isFinite(storedModel.metrics.testLoss) &&
      Math.abs(storedModel.metrics.testAccuracy * 100 - parseFloat(trained.test)) <= 0.051 &&
      Math.abs(storedModel.metrics.drawingAccuracy * 100 - parseFloat(trained.drawing)) <= 0.051 &&
      Math.abs(storedModel.metrics.testLoss - Number(trained.loss)) <= 0.00051,
    "IndexedDB record did not exactly contain the validated C serialization and metadata",
    storedModel,
  );

  const predictionsBeforeReload = await evaluate(
    devtools,
    sessionId,
    fixturePredictionsExpression,
  );
  check(
    predictionsBeforeReload.every((fixture) =>
      fixture.label >= 0 && fixture.label <= 9 &&
      fixture.prediction >= 0 && fixture.prediction <= 9 &&
      fixture.probabilities.length === 10 &&
      fixture.probabilities.every(Number.isFinite) &&
      Math.abs(fixture.probabilities.reduce((sum, value) => sum + value, 0) - 1) < 1e-9),
    "Pre-reload C prediction fixtures were invalid",
    predictionsBeforeReload,
  );

  await devtools.call("Page.reload", { ignoreCache: true }, sessionId);
  const restoredReady = await waitFor(
    devtools,
    sessionId,
    readyExpression,
    (value) => value.runtime === "Saved model loaded" || value.runtime === "Load failed",
    TRAINING_DEADLINE_MS,
    "Saved-model reload",
  );
  check(
    restoredReady.runtime === "Saved model loaded" && restoredReady.origin === expectedOrigin &&
      !restoredReady.disabled && !restoredReady.resetDisabled,
    "Same-origin/profile reload did not restore the saved model",
    restoredReady,
  );

  const restored = await evaluate(
    devtools,
    sessionId,
    `(() => {
      const canvas = document.querySelector("#network-canvas");
      const bounds = canvas.getBoundingClientRect();
      window.__enenUnifiedIdentity = { canvas, renderer: networkRenderer };
      return {
        storedModelLoaded: state.storedModelLoaded,
        modelReady: state.modelReady,
        modelTrained: state.modelTrained,
        configured: state.configured,
        training: state.training,
        activeTab: state.activeTab,
        shape: modelShape(),
        epochsTrained: state.wasm._web_epochs_trained(),
        currentConfig: state.currentConfig,
        architecture: document.querySelector("#architecture").value,
        epochs: document.querySelector("#epochs").value,
        epochOutput: document.querySelector("#epochs-value").value,
        learningRate: document.querySelector("#learning-rate").value,
        epochLabel: document.querySelector("#epoch-label").textContent,
        test: document.querySelector("#test-accuracy").textContent,
        loss: document.querySelector("#test-loss").textContent,
        drawing: document.querySelector("#drawing-accuracy").textContent,
        storageStatus: document.querySelector("#storage-status").textContent,
        trainingStatus: document.querySelector("#training-status").textContent,
        setupHidden: document.querySelector("#setup-view").hidden,
        stageHidden: document.querySelector("#training-stage").hidden,
        locked: document.querySelector("#recognition-panel").getAttribute("aria-disabled"),
        drawingDisabled: document.querySelector("#drawing-canvas").getAttribute("aria-disabled"),
        drawToolDisabled: document.querySelector("#draw-tool-button").disabled,
        eraserToolDisabled: document.querySelector("#eraser-tool-button").disabled,
        drawPressed: document.querySelector("#draw-tool-button").getAttribute("aria-pressed"),
        eraserPressed: document.querySelector("#eraser-tool-button").getAttribute("aria-pressed"),
        drawingTool: state.drawingTool,
        canvasTool: document.querySelector("#drawing-canvas").dataset.tool,
        clearDisabled: document.querySelector("#clear-button").disabled,
        exampleDisabled: document.querySelector("#example-button").disabled,
        networkCount: document.querySelectorAll("#network-canvas").length,
        networkVisible: bounds.width > 0 && bounds.height > 0 &&
          getComputedStyle(canvas).display !== "none",
        networkShape: networkRenderer?.snapshot?.layerSizes || [],
        networkStatus: document.querySelector("#network-status").textContent,
      };
    })()`,
  );
  check(
    restored.storedModelLoaded && restored.modelReady && restored.modelTrained &&
      restored.configured && !restored.training && restored.activeTab === "train" &&
      restored.shape.join(",") === "64,48,24,10" && restored.epochsTrained === 0 &&
      restored.currentConfig.firstHidden === 48 &&
      restored.currentConfig.secondHidden === 24 &&
      restored.currentConfig.epochs === 25 &&
      restored.currentConfig.learningRate === 0.05 &&
      restored.architecture === "48,24" && restored.epochs === "25" &&
      restored.epochOutput === "25" && restored.learningRate === "0.05" &&
      restored.epochLabel === "Saved after 25 epochs" &&
      restored.test === trained.test && restored.loss === trained.loss &&
      restored.drawing === trained.drawing && restored.storageStatus.includes("Saved locally") &&
      restored.trainingStatus.includes("Loaded the last validated model") &&
      !restored.setupHidden && restored.stageHidden && restored.locked === "false" &&
      restored.drawingDisabled === "false" && !restored.drawToolDisabled &&
      !restored.eraserToolDisabled && restored.drawPressed === "true" &&
      restored.eraserPressed === "false" && restored.drawingTool === "draw" &&
      restored.canvasTool === "draw" && !restored.clearDisabled &&
      !restored.exampleDisabled && restored.networkCount === 1 && restored.networkVisible &&
      restored.networkShape.join(",") === "64,48,24,10" &&
      restored.networkStatus.includes("Saved model"),
    "Reload did not restore exact controls, metrics, shape, and unlocked state without training",
    restored,
  );
  checkTabState(await evaluate(devtools, sessionId, tabStateExpression), "train");

  const restoredModel = await evaluate(devtools, sessionId, modelAuditExpression);
  check(
    restoredModel.exact && restoredModel.saved === 0 &&
      restoredModel.storedLength === storedModel.storedLength &&
      restoredModel.serializedLength === storedModel.serializedLength &&
      restoredModel.storedHash === storedModel.storedHash &&
      restoredModel.serializedHash === storedModel.serializedHash &&
      restoredModel.revision === storedModel.revision &&
      restoredModel.savedAt === storedModel.savedAt &&
      JSON.stringify(restoredModel.config) === JSON.stringify(storedModel.config) &&
      JSON.stringify(restoredModel.metrics) === JSON.stringify(storedModel.metrics),
    "Reloaded C model bytes or IndexedDB metadata changed",
    { before: storedModel, after: restoredModel },
  );
  const predictionsAfterReload = await evaluate(
    devtools,
    sessionId,
    fixturePredictionsExpression,
  );
  check(
    JSON.stringify(predictionsAfterReload) === JSON.stringify(predictionsBeforeReload),
    "Reload changed deterministic C predictions or probabilities",
    { before: predictionsBeforeReload, after: predictionsAfterReload },
  );

  await evaluate(
    devtools,
    sessionId,
    `(() => {
      const wasm = state.wasm;
      const inspectInput = wasm._web_inspect_input;
      const audit = window.__enenRecognitionAudit = { calls: [] };
      wasm._web_inspect_input = () => {
        const result = inspectInput();
        const layerSizes = modelShape();
        const activations = [];
        for (let layer = 0; layer < layerSizes.length; layer += 1) {
          for (let node = 0; node < layerSizes[layer]; node += 1) {
            activations.push(wasm._web_activation(layer, node));
          }
        }
        const inputOffset = wasm._web_input_buffer() / Float64Array.BYTES_PER_ELEMENT;
        audit.calls.push({
          result,
          layerSizes,
          activations,
          input: Array.from(wasm.HEAPF64.subarray(inputOffset, inputOffset + 64)),
        });
        return result;
      };
      window.__enenCanvasAudit = () => {
        const rasterStats = (canvas) => {
          const context = canvas.getContext("2d");
          const rgba = context.getImageData(0, 0, canvas.width, canvas.height).data;
          let alphaSum = 0;
          let alphaPixels = 0;
          let hash = 2166136261;
          for (let index = 0; index < rgba.length; index += 4) {
            const alpha = rgba[index + 3];
            alphaSum += alpha;
            if (alpha > 10) alphaPixels += 1;
            hash = Math.imul(hash ^ alpha, 16777619);
          }
          return { alphaSum, alphaPixels, hash: hash >>> 0 };
        };
        const network = document.querySelector("#network-canvas");
        const networkRgba = network.getContext("2d").getImageData(
          0, 0, network.width, network.height
        ).data;
        let networkHash = 2166136261;
        for (let index = 0; index < networkRgba.length; index += 16) {
          networkHash = Math.imul(networkHash ^ networkRgba[index], 16777619);
          networkHash = Math.imul(networkHash ^ networkRgba[index + 1], 16777619);
          networkHash = Math.imul(networkHash ^ networkRgba[index + 2], 16777619);
          networkHash = Math.imul(networkHash ^ networkRgba[index + 3], 16777619);
        }
        const normalizedPixels = canvasToPixels(drawingMemoryCanvas);
        const normalized = normalizedPixels ? Array.from(normalizedPixels) : Array(64).fill(0);
        const snapshot = networkRenderer.snapshot;
        let rendererInputDelta = 0;
        for (let index = 0; index < 64; index += 1) {
          rendererInputDelta = Math.max(
            rendererInputDelta,
            Math.abs(snapshot.activations[index] - normalized[index]),
          );
        }
        return {
          visible: rasterStats(ui.drawingCanvas),
          memory: rasterStats(drawingMemoryCanvas),
          normalized,
          probabilities: Array.from(
            { length: 10 },
            (_, digit) => state.wasm._web_probability(digit),
          ),
          activations: Array.from(snapshot.activations),
          rendererInputDelta,
          renderCount: Number(network.dataset.renderCount || 0),
          version: Number(network.dataset.activationVersion || 0),
          networkHash: networkHash >>> 0,
          inspectCalls: window.__enenRecognitionAudit.calls.length,
          prediction: document.querySelector("#prediction-digit").textContent,
          announcement: document.querySelector("#prediction-announcement").textContent,
          networkStatus: document.querySelector("#network-status").textContent,
          hasInk: state.hasInk,
          drawing: state.drawing,
          drawingTool: state.drawingTool,
          canvasTool: ui.drawingCanvas.dataset.tool,
          canvasLabel: ui.drawingCanvas.getAttribute("aria-label"),
          drawDisabled: ui.drawToolButton.disabled,
          eraserDisabled: ui.eraserToolButton.disabled,
          drawPressed: ui.drawToolButton.getAttribute("aria-pressed"),
          eraserPressed: ui.eraserToolButton.getAttribute("aria-pressed"),
          erasingClass: ui.drawingCanvas.classList.contains("is-erasing"),
        };
      };
      document.querySelector("#recognize-tab").click();
      const canvas = document.querySelector("#network-canvas");
      const context = canvas.getContext("2d");
      const rgba = context.getImageData(0, 0, canvas.width, canvas.height).data;
      let hash = 2166136261;
      for (let index = 0; index < rgba.length; index += 16) {
        hash = Math.imul(hash ^ rgba[index], 16777619);
        hash = Math.imul(hash ^ rgba[index + 1], 16777619);
        hash = Math.imul(hash ^ rgba[index + 2], 16777619);
      }
      window.__enenRestingNetwork = {
        renderCount: Number(canvas.dataset.renderCount || 0),
        version: Number(canvas.dataset.activationVersion || 0),
        hash: hash >>> 0,
      };
    })()`,
  );
  checkTabState(await evaluate(devtools, sessionId, tabStateExpression), "recognize");
  const recognizeNetwork = await evaluate(devtools, sessionId, networkStateExpression);
  check(
    recognizeNetwork.count === 1 && recognizeNetwork.visible && recognizeNetwork.rendererSame &&
      recognizeNetwork.shape.join(",") === "64,48,24,10",
    "Recognize mode hid or replaced the unified restored-model monitor",
    recognizeNetwork,
  );

  await evaluate(devtools, sessionId, "document.querySelector('#example-button').click()");
  await waitFor(
    devtools,
    sessionId,
    `(() => ({
      renderCount: Number(document.querySelector("#network-canvas").dataset.renderCount || 0),
      version: Number(document.querySelector("#network-canvas").dataset.activationVersion || 0),
      inspectCalls: window.__enenRecognitionAudit.calls.length
    }))()`,
    (value) => value.renderCount > 0 &&
      value.renderCount > recognizeNetwork.renderCount && value.inspectCalls > 0,
    STARTUP_DEADLINE_MS,
    "Unified recognition activation snapshot",
  );
  await delay(250);
  const recognitionFrame = await evaluate(
    devtools,
    sessionId,
    `(() => {
      const canvas = document.querySelector("#network-canvas");
      const snapshot = networkRenderer?.snapshot;
      const centeredPixels = canvasToPixels(drawingMemoryCanvas);
      const inspectCall = window.__enenRecognitionAudit.calls.at(-1);
      const rgba = canvas.getContext("2d").getImageData(
        0, 0, canvas.width, canvas.height
      ).data;
      let canvasHash = 2166136261;
      let nontransparentPixels = 0;
      for (let index = 0; index < rgba.length; index += 16) {
        if (rgba[index + 3] > 0) nontransparentPixels += 1;
        canvasHash = Math.imul(canvasHash ^ rgba[index], 16777619);
        canvasHash = Math.imul(canvasHash ^ rgba[index + 1], 16777619);
        canvasHash = Math.imul(canvasHash ^ rgba[index + 2], 16777619);
        canvasHash = Math.imul(canvasHash ^ rgba[index + 3], 16777619);
      }
      let inputDelta = 0;
      let inspectedInputDelta = 0;
      let inspectedSnapshotDelta = 0;
      for (let index = 0; index < 64; index += 1) {
        inputDelta = Math.max(
          inputDelta,
          Math.abs(snapshot.activations[index] - centeredPixels[index]),
        );
        inspectedInputDelta = Math.max(
          inspectedInputDelta,
          Math.abs(inspectCall.input[index] - centeredPixels[index]),
        );
      }
      for (let index = 0; index < snapshot.activations.length; index += 1) {
        inspectedSnapshotDelta = Math.max(
          inspectedSnapshotDelta,
          Math.abs(inspectCall.activations[index] - snapshot.activations[index]),
        );
      }
      const bounds = canvas.getBoundingClientRect();
      return {
        renderCount: Number(canvas.dataset.renderCount || 0),
        version: Number(canvas.dataset.activationVersion || 0),
        snapshotVersion: snapshot.version,
        stateVersion: state.networkVisualizationVersion,
        shape: snapshot.layerSizes,
        inspectShape: inspectCall.layerSizes,
        activationCount: snapshot.activations.length,
        activationsFinite: snapshot.activations.every(Number.isFinite),
        activationMinimum: Math.min(...snapshot.activations),
        activationMaximum: Math.max(...snapshot.activations),
        outputSum: snapshot.activations.slice(-10).reduce(
          (sum, activation) => sum + activation,
          0,
        ),
        inspectResult: inspectCall.result,
        inspectCalls: window.__enenRecognitionAudit.calls.length,
        inputDelta,
        inspectedInputDelta,
        inspectedSnapshotDelta,
        canvasHash: canvasHash >>> 0,
        nontransparentPixels,
        prediction: document.querySelector("#prediction-digit").textContent,
        status: document.querySelector("#network-status").textContent,
        canvasCount: document.querySelectorAll("#network-canvas").length,
        visible: bounds.width > 0 && bounds.height > 0,
        rendererSame: window.__enenUnifiedIdentity.renderer === networkRenderer &&
          window.__enenUnifiedIdentity.canvas === canvas,
      };
    })()`,
  );
  check(
    recognitionFrame.renderCount > recognizeNetwork.renderCount &&
      recognitionFrame.version > recognizeNetwork.version &&
      recognitionFrame.version === recognitionFrame.snapshotVersion &&
      recognitionFrame.version === recognitionFrame.stateVersion &&
      recognitionFrame.shape.join(",") === "64,48,24,10" &&
      recognitionFrame.inspectShape.join(",") === "64,48,24,10" &&
      recognitionFrame.activationCount === 146 && recognitionFrame.activationsFinite &&
      recognitionFrame.activationMinimum >= 0 && recognitionFrame.activationMaximum <= 1 &&
      Math.abs(recognitionFrame.outputSum - 1) < 1e-9 &&
      recognitionFrame.inspectResult >= 0 && recognitionFrame.inspectResult <= 9 &&
      recognitionFrame.inspectCalls > 0 && recognitionFrame.inputDelta < 1e-12 &&
      recognitionFrame.inspectedInputDelta < 1e-12 &&
      recognitionFrame.inspectedSnapshotDelta < 1e-12 &&
      recognitionFrame.nontransparentPixels > 0 &&
      recognitionFrame.canvasHash !== (await evaluate(
        devtools,
        sessionId,
        "window.__enenRestingNetwork.hash",
      )) &&
      /^[0-9]$/.test(recognitionFrame.prediction) &&
      recognitionFrame.status.includes("Recognizing") && recognitionFrame.canvasCount === 1 &&
      recognitionFrame.visible && recognitionFrame.rendererSame,
    "Unified recognition visualization was not backed by centered C activations",
    recognitionFrame,
  );

  const beforeErase = await evaluate(devtools, sessionId, "window.__enenCanvasAudit()");
  check(
    beforeErase.visible.alphaSum > 0 && beforeErase.memory.alphaSum > 0 &&
      beforeErase.hasInk && !beforeErase.drawing && beforeErase.drawingTool === "draw" &&
      beforeErase.canvasTool === "draw" && beforeErase.drawPressed === "true" &&
      beforeErase.eraserPressed === "false" && !beforeErase.drawDisabled &&
      !beforeErase.eraserDisabled && beforeErase.rendererInputDelta < 1e-12 &&
      /^[0-9]$/.test(beforeErase.prediction),
    "Draw example did not begin the eraser regression in an inked Draw state",
    beforeErase,
  );

  const erasePath = await evaluate(
    devtools,
    sessionId,
    `(() => {
      const canvas = document.querySelector("#drawing-canvas");
      const rgba = canvas.getContext("2d").getImageData(
        0, 0, canvas.width, canvas.height
      ).data;
      const rowSums = new Float64Array(canvas.height);
      const columnSums = new Float64Array(canvas.width);
      for (let y = 0; y < canvas.height; y += 1) {
        for (let x = 0; x < canvas.width; x += 1) {
          const alpha = rgba[(y * canvas.width + x) * 4 + 3];
          rowSums[y] += alpha;
          columnSums[x] += alpha;
        }
      }
      let row = 0;
      let column = 0;
      for (let index = 1; index < rowSums.length; index += 1) {
        if (rowSums[index] > rowSums[row]) row = index;
      }
      for (let index = 1; index < columnSums.length; index += 1) {
        if (columnSums[index] > columnSums[column]) column = index;
      }
      const bounds = canvas.getBoundingClientRect();
      const inset = 12;
      const horizontal = rowSums[row] >= columnSums[column];
      const toClient = (x, y) => ({
        x: bounds.left + x / canvas.width * bounds.width,
        y: bounds.top + y / canvas.height * bounds.height,
      });
      const start = horizontal
        ? toClient(inset, row)
        : toClient(column, inset);
      const end = horizontal
        ? toClient(canvas.width - inset, row)
        : toClient(column, canvas.height - inset);
      document.querySelector("#eraser-tool-button").click();
      return {
        start,
        end,
        horizontal,
        row,
        column,
        density: horizontal ? rowSums[row] : columnSums[column],
        drawingTool: state.drawingTool,
        canvasTool: canvas.dataset.tool,
        drawPressed: document.querySelector("#draw-tool-button").getAttribute("aria-pressed"),
        eraserPressed: document.querySelector("#eraser-tool-button").getAttribute("aria-pressed"),
        erasingClass: canvas.classList.contains("is-erasing"),
        canvasLabel: canvas.getAttribute("aria-label"),
        message: document.querySelector("#canvas-message").textContent,
      };
    })()`,
  );
  check(
    erasePath.density > 0 && erasePath.drawingTool === "erase" &&
      erasePath.canvasTool === "erase" && erasePath.drawPressed === "false" &&
      erasePath.eraserPressed === "true" && erasePath.erasingClass &&
      erasePath.canvasLabel.startsWith("Erase") && erasePath.message.includes("erase ink"),
    "Eraser button did not expose the selected erasing state",
    erasePath,
  );

  await dispatchPointerDrag(devtools, sessionId, erasePath.start, erasePath.end, 24);
  await waitFor(
    devtools,
    sessionId,
    `(() => ({
      drawing: state.drawing,
      hasInk: state.hasInk,
      tool: state.drawingTool,
      inspectCalls: window.__enenRecognitionAudit.calls.length,
      renderCount: Number(document.querySelector("#network-canvas").dataset.renderCount || 0),
      version: Number(document.querySelector("#network-canvas").dataset.activationVersion || 0)
    }))()`,
    (value) => !value.drawing && value.hasInk && value.tool === "erase" &&
      value.inspectCalls > beforeErase.inspectCalls &&
      value.renderCount > beforeErase.renderCount && value.version > beforeErase.version,
    STARTUP_DEADLINE_MS,
    "Partial eraser recognition refresh",
  );
  await delay(250);
  const afterErase = await evaluate(devtools, sessionId, "window.__enenCanvasAudit()");
  const erasedPixelDelta = maximumDelta(beforeErase.normalized, afterErase.normalized);
  const erasedProbabilityDelta = maximumDelta(
    beforeErase.probabilities,
    afterErase.probabilities,
  );
  const erasedActivationDelta = maximumDelta(
    beforeErase.activations,
    afterErase.activations,
  );
  check(
    afterErase.visible.alphaSum < beforeErase.visible.alphaSum * 0.95 &&
      afterErase.memory.alphaSum < beforeErase.memory.alphaSum * 0.95 &&
      afterErase.visible.alphaSum > 0 && afterErase.memory.alphaSum > 0 &&
      afterErase.visible.hash !== beforeErase.visible.hash &&
      afterErase.memory.hash !== beforeErase.memory.hash && afterErase.hasInk &&
      afterErase.drawingTool === "erase" && afterErase.canvasTool === "erase" &&
      afterErase.drawPressed === "false" && afterErase.eraserPressed === "true" &&
      afterErase.erasingClass && afterErase.inspectCalls > beforeErase.inspectCalls &&
      afterErase.renderCount > beforeErase.renderCount && afterErase.version > beforeErase.version &&
      afterErase.networkHash !== beforeErase.networkHash &&
      erasedPixelDelta > 1e-5 && erasedProbabilityDelta > 1e-10 &&
      erasedActivationDelta > 1e-8 && afterErase.rendererInputDelta < 1e-12 &&
      /^[0-9]$/.test(afterErase.prediction),
    "Trusted eraser drag did not remove visible/memory ink and refresh C inference",
    {
      before: beforeErase,
      after: afterErase,
      erasedPixelDelta,
      erasedProbabilityDelta,
      erasedActivationDelta,
      path: erasePath,
    },
  );

  const drawSelected = await evaluate(
    devtools,
    sessionId,
    `(() => {
      document.querySelector("#draw-tool-button").click();
      return {
        drawingTool: state.drawingTool,
        canvasTool: document.querySelector("#drawing-canvas").dataset.tool,
        drawPressed: document.querySelector("#draw-tool-button").getAttribute("aria-pressed"),
        eraserPressed: document.querySelector("#eraser-tool-button").getAttribute("aria-pressed"),
        erasingClass: document.querySelector("#drawing-canvas").classList.contains("is-erasing"),
        label: document.querySelector("#drawing-canvas").getAttribute("aria-label"),
      };
    })()`,
  );
  check(
    drawSelected.drawingTool === "draw" && drawSelected.canvasTool === "draw" &&
      drawSelected.drawPressed === "true" && drawSelected.eraserPressed === "false" &&
      !drawSelected.erasingClass && drawSelected.label.startsWith("Draw"),
    "Draw button did not restore the drawing tool state",
    drawSelected,
  );
  await dispatchPointerDrag(devtools, sessionId, erasePath.start, erasePath.end, 24);
  await waitFor(
    devtools,
    sessionId,
    `(() => ({
      drawing: state.drawing,
      inspectCalls: window.__enenRecognitionAudit.calls.length,
      renderCount: Number(document.querySelector("#network-canvas").dataset.renderCount || 0),
      version: Number(document.querySelector("#network-canvas").dataset.activationVersion || 0)
    }))()`,
    (value) => !value.drawing && value.inspectCalls > afterErase.inspectCalls &&
      value.renderCount > afterErase.renderCount && value.version > afterErase.version,
    STARTUP_DEADLINE_MS,
    "Draw-after-erase recognition refresh",
  );
  await delay(250);
  const afterDraw = await evaluate(devtools, sessionId, "window.__enenCanvasAudit()");
  check(
    afterDraw.visible.alphaSum > afterErase.visible.alphaSum * 1.05 &&
      afterDraw.memory.alphaSum > afterErase.memory.alphaSum * 1.05 &&
      afterDraw.visible.hash !== afterErase.visible.hash &&
      afterDraw.memory.hash !== afterErase.memory.hash && afterDraw.hasInk &&
      afterDraw.drawingTool === "draw" && afterDraw.canvasTool === "draw" &&
      afterDraw.drawPressed === "true" && afterDraw.eraserPressed === "false" &&
      afterDraw.inspectCalls > afterErase.inspectCalls &&
      afterDraw.renderCount > afterErase.renderCount && afterDraw.version > afterErase.version &&
      afterDraw.networkHash !== afterErase.networkHash &&
      maximumDelta(afterErase.normalized, afterDraw.normalized) > 1e-5 &&
      maximumDelta(afterErase.activations, afterDraw.activations) > 1e-8 &&
      afterDraw.rendererInputDelta < 1e-12 && /^[0-9]$/.test(afterDraw.prediction),
    "Switching back to Draw did not add ink and refresh the unified C network",
    { erased: afterErase, drawn: afterDraw },
  );

  const cleared = await evaluate(
    devtools,
    sessionId,
    `(() => {
      document.querySelector("#eraser-tool-button").click();
      document.querySelector("#clear-button").click();
      return window.__enenCanvasAudit();
    })()`,
  );
  check(
    cleared.visible.alphaSum === 0 && cleared.memory.alphaSum === 0 &&
      cleared.normalized.every((value) => value === 0) &&
      cleared.prediction === "—" && !cleared.hasInk &&
      cleared.networkStatus.includes("Trained weights") &&
      Math.max(...cleared.activations.slice(0, 64)) === 0 &&
      cleared.rendererInputDelta === 0 && cleared.drawingTool === "draw" &&
      cleared.canvasTool === "draw" && cleared.drawPressed === "true" &&
      cleared.eraserPressed === "false" && !cleared.erasingClass &&
      cleared.announcement === "Canvas cleared." &&
      cleared.renderCount > afterDraw.renderCount && cleared.version > afterDraw.version,
    "Clear did not empty both canvases, reset Draw, announce, and rest the network",
    cleared,
  );

  const dotPath = await evaluate(
    devtools,
    sessionId,
    `(() => {
      const bounds = document.querySelector("#drawing-canvas").getBoundingClientRect();
      return {
        start: { x: bounds.left + bounds.width / 2 - 1, y: bounds.top + bounds.height / 2 },
        end: { x: bounds.left + bounds.width / 2 + 1, y: bounds.top + bounds.height / 2 },
      };
    })()`,
  );
  await dispatchPointerDrag(devtools, sessionId, dotPath.start, dotPath.end, 2);
  await waitFor(
    devtools,
    sessionId,
    `(() => ({
      hasInk: state.hasInk,
      drawing: state.drawing,
      inspectCalls: window.__enenRecognitionAudit.calls.length
    }))()`,
    (value) => value.hasInk && !value.drawing && value.inspectCalls > cleared.inspectCalls,
    STARTUP_DEADLINE_MS,
    "Final-ink eraser fixture",
  );
  await delay(200);
  const dot = await evaluate(devtools, sessionId, "window.__enenCanvasAudit()");
  check(
    dot.visible.alphaSum > 0 && dot.memory.alphaSum > 0 && dot.hasInk &&
      dot.drawingTool === "draw" && dot.rendererInputDelta < 1e-12,
    "Draw tool did not create the final-ink eraser fixture",
    dot,
  );
  const synchronousFinalErase = await evaluate(
    devtools,
    sessionId,
    `(() => {
      const canvas = document.querySelector("#drawing-canvas");
      const bounds = canvas.getBoundingClientRect();
      const pointerId = 9173;
      const centerY = bounds.top + bounds.height / 2;
      const eventAt = (clientX) => ({
        pointerId,
        button: 0,
        clientX,
        clientY: centerY,
        preventDefault() {},
        getCoalescedEvents() { return []; },
      });
      const originalSetPointerCapture = canvas.setPointerCapture;
      const originalReleasePointerCapture = canvas.releasePointerCapture;
      canvas.setPointerCapture = () => {};
      canvas.releasePointerCapture = () => {};
      try {
        document.querySelector("#eraser-tool-button").click();
        beginDrawing(eventAt(bounds.left + bounds.width / 2 - 30));
        continueDrawing(eventAt(bounds.left + bounds.width / 2 + 30));
        const queuedBeforeEnd = state.recognitionFrame !== 0;
        endDrawing(eventAt(bounds.left + bounds.width / 2 + 30));
        return {
          queuedBeforeEnd,
          queuedAfterEnd: state.recognitionFrame,
          hasInk: state.hasInk,
          drawing: state.drawing,
          drawingTool: state.drawingTool,
          announcement: document.querySelector("#prediction-announcement").textContent,
        };
      } finally {
        canvas.setPointerCapture = originalSetPointerCapture;
        canvas.releasePointerCapture = originalReleasePointerCapture;
      }
    })()`,
  );
  check(
    synchronousFinalErase.queuedBeforeEnd && synchronousFinalErase.queuedAfterEnd === 0 &&
      !synchronousFinalErase.hasInk && !synchronousFinalErase.drawing &&
      synchronousFinalErase.drawingTool === "draw" &&
      synchronousFinalErase.announcement === "Canvas empty.",
    "Same-task final erase did not cancel its queued recognition frame",
    synchronousFinalErase,
  );
  await delay(100);
  const emptyAfterErase = await evaluate(devtools, sessionId, "window.__enenCanvasAudit()");
  check(
    emptyAfterErase.visible.alphaSum === 0 && emptyAfterErase.memory.alphaSum === 0 &&
      emptyAfterErase.normalized.every((value) => value === 0) &&
      !emptyAfterErase.hasInk && emptyAfterErase.prediction === "—" &&
      emptyAfterErase.announcement === "Canvas empty." &&
      emptyAfterErase.networkStatus.includes("Trained weights") &&
      Math.max(...emptyAfterErase.activations.slice(0, 64)) === 0 &&
      emptyAfterErase.drawingTool === "draw" && emptyAfterErase.canvasTool === "draw" &&
      emptyAfterErase.drawPressed === "true" && emptyAfterErase.eraserPressed === "false" &&
      !emptyAfterErase.erasingClass && emptyAfterErase.renderCount > dot.renderCount &&
      emptyAfterErase.version > dot.version,
    "Erasing final ink did not reset Draw, announce, and rest both canvases/network",
    emptyAfterErase,
  );

  const edgePaths = await evaluate(
    devtools,
    sessionId,
    `(() => {
      const bounds = document.querySelector("#drawing-canvas").getBoundingClientRect();
      const y = bounds.top + bounds.height * 0.68;
      return {
        outwardDraw: {
          start: { x: bounds.left + 2, y },
          end: { x: bounds.left - 50, y },
        },
        visibleEdgeErase: {
          start: { x: bounds.left + 8, y },
          end: { x: bounds.left, y },
        },
      };
    })()`,
  );
  await dispatchPointerDrag(
    devtools,
    sessionId,
    edgePaths.outwardDraw.start,
    edgePaths.outwardDraw.end,
    10,
  );
  await waitFor(
    devtools,
    sessionId,
    `(() => ({
      hasInk: state.hasInk,
      drawing: state.drawing,
      inspectCalls: window.__enenRecognitionAudit.calls.length
    }))()`,
    (value) => value.hasInk && !value.drawing &&
      value.inspectCalls > emptyAfterErase.inspectCalls,
    STARTUP_DEADLINE_MS,
    "Outward edge drawing stroke",
  );
  await delay(200);
  const edgeDrawn = await evaluate(devtools, sessionId, "window.__enenCanvasAudit()");
  check(
    edgeDrawn.visible.alphaSum > 0 &&
      edgeDrawn.memory.alphaSum > edgeDrawn.visible.alphaSum && edgeDrawn.hasInk &&
      edgeDrawn.drawingTool === "draw" && edgeDrawn.rendererInputDelta < 1e-12,
    "Outward edge stroke did not exercise brush-radius ink in padded memory",
    { paths: edgePaths, drawn: edgeDrawn },
  );

  await evaluate(
    devtools,
    sessionId,
    "document.querySelector('#eraser-tool-button').click()",
  );
  await dispatchPointerDrag(
    devtools,
    sessionId,
    edgePaths.visibleEdgeErase.start,
    edgePaths.visibleEdgeErase.end,
    4,
  );
  await waitFor(
    devtools,
    sessionId,
    `(() => ({
      hasInk: state.hasInk,
      drawing: state.drawing,
      drawingTool: state.drawingTool,
      announcement: document.querySelector("#prediction-announcement").textContent
    }))()`,
    (value) => !value.hasInk && !value.drawing && value.drawingTool === "draw" &&
      value.announcement === "Canvas empty.",
    STARTUP_DEADLINE_MS,
    "Outward edge erase",
  );
  await delay(100);
  const edgeErased = await evaluate(devtools, sessionId, "window.__enenCanvasAudit()");
  check(
    edgeErased.visible.alphaSum === 0 && edgeErased.memory.alphaSum === 0 &&
      edgeErased.normalized.every((value) => value === 0) && !edgeErased.hasInk &&
      edgeErased.prediction === "—" && edgeErased.announcement === "Canvas empty." &&
      edgeErased.drawingTool === "draw" && edgeErased.canvasTool === "draw" &&
      edgeErased.drawPressed === "true" && edgeErased.eraserPressed === "false" &&
      edgeErased.networkStatus.includes("Trained weights") &&
      Math.max(...edgeErased.activations.slice(0, 64)) === 0,
    "Visible-empty edge erase left ghost ink in padded memory or app state",
    edgeErased,
  );

  await dispatchPointerDrag(
    devtools,
    sessionId,
    edgePaths.outwardDraw.start,
    edgePaths.outwardDraw.end,
    10,
  );
  await waitFor(
    devtools,
    sessionId,
    `(() => ({ hasInk: state.hasInk, drawing: state.drawing }))()`,
    (value) => value.hasInk && !value.drawing,
    STARTUP_DEADLINE_MS,
    "Outward edge clear fixture",
  );
  const edgeBeforeClear = await evaluate(devtools, sessionId, "window.__enenCanvasAudit()");
  check(
    edgeBeforeClear.visible.alphaSum > 0 &&
      edgeBeforeClear.memory.alphaSum > edgeBeforeClear.visible.alphaSum,
    "Second outward edge stroke did not reach padded memory",
    edgeBeforeClear,
  );
  const edgeCleared = await evaluate(
    devtools,
    sessionId,
    `(() => {
      document.querySelector("#clear-button").click();
      return window.__enenCanvasAudit();
    })()`,
  );
  check(
    edgeCleared.visible.alphaSum === 0 && edgeCleared.memory.alphaSum === 0 &&
      edgeCleared.normalized.every((value) => value === 0) && !edgeCleared.hasInk &&
      edgeCleared.prediction === "—" && edgeCleared.announcement === "Canvas cleared." &&
      edgeCleared.drawingTool === "draw" && edgeCleared.canvasTool === "draw" &&
      edgeCleared.networkStatus.includes("Trained weights") &&
      Math.max(...edgeCleared.activations.slice(0, 64)) === 0,
    "Clear left outward-edge ghost ink in padded memory or app state",
    edgeCleared,
  );

  const centering = await evaluate(
    devtools,
    sessionId,
    `(() => {
      const positions = [
        [140, 140],
        [45, 45],
        [235, 45],
        [45, 235],
        [235, 235],
      ];
      let maxPixelDelta = 0;
      let maxProbabilityDelta = 0;
      let fixtureMaxPixelDelta = 0;
      const failures = [];

      function rasterizeAt(digit, x, y) {
        drawingMemoryContext.clearRect(
          0, 0, drawingMemoryCanvas.width, drawingMemoryCanvas.height
        );
        drawingMemoryContext.save();
        drawingMemoryContext.translate(
          x + DRAWING_MEMORY_PADDING,
          y + DRAWING_MEMORY_PADDING,
        );
        drawingMemoryContext.scale(1.5, 1.5);
        drawingMemoryContext.lineWidth = 8;
        drawingMemoryContext.strokeStyle = "#fff";
        drawVectorDigit(drawingMemoryContext, digit);
        drawingMemoryContext.restore();
        const pixels = canvasToPixels(drawingMemoryCanvas);
        writeWasmInput(pixels);
        const prediction = state.wasm._web_predict();
        const probabilities = Array.from(
          { length: 10 },
          (_, candidate) => state.wasm._web_probability(candidate),
        );
        return { pixels: Array.from(pixels), prediction, probabilities };
      }

      for (let digit = 0; digit < 10; digit += 1) {
        const centered = rasterizeAt(digit, ...positions[0]);
        if (centered.prediction !== digit) {
          failures.push("centered " + digit + " predicted " + centered.prediction);
        }
        for (const position of positions.slice(1)) {
          const shifted = rasterizeAt(digit, ...position);
          if (shifted.prediction !== centered.prediction) {
            failures.push(
              "digit " + digit + " changed to " + shifted.prediction +
                " at " + position.join(","),
            );
          }
          for (let index = 0; index < 64; index += 1) {
            maxPixelDelta = Math.max(
              maxPixelDelta,
              Math.abs(shifted.pixels[index] - centered.pixels[index]),
            );
          }
          for (let candidate = 0; candidate < 10; candidate += 1) {
            maxProbabilityDelta = Math.max(
              maxProbabilityDelta,
              Math.abs(
                shifted.probabilities[candidate] - centered.probabilities[candidate]
              ),
            );
          }
        }
      }

      const fixtureCanvas = document.createElement("canvas");
      fixtureCanvas.width = 280;
      fixtureCanvas.height = 280;
      renderDrawingCheckDigit(
        fixtureCanvas,
        1,
        new SeededRandom(0x6f665386),
        true,
      );
      const fixtureOffsets = [0, -96, -50, -25, 25, 50, 75, 100, 125, 129];
      let fixtureCentered = null;
      let fixturePrediction = null;
      for (const offset of fixtureOffsets) {
        drawingMemoryContext.clearRect(
          0, 0, drawingMemoryCanvas.width, drawingMemoryCanvas.height
        );
        drawingMemoryContext.drawImage(
          fixtureCanvas,
          DRAWING_MEMORY_PADDING + offset,
          DRAWING_MEMORY_PADDING,
        );
        const pixels = canvasToPixels(drawingMemoryCanvas);
        if (offset === 0) fixtureCentered = pixels;
        writeWasmInput(pixels);
        const predicted = state.wasm._web_predict();
        if (offset === 0) fixturePrediction = predicted;
        else if (predicted !== fixturePrediction) {
          failures.push(
            "translated 1 changed from " + fixturePrediction + " to " +
              predicted + " at " + offset,
          );
        }
        if (fixtureCentered) {
          for (let index = 0; index < 64; index += 1) {
            fixtureMaxPixelDelta = Math.max(
              fixtureMaxPixelDelta,
              Math.abs(pixels[index] - fixtureCentered[index]),
            );
          }
        }
      }
      return {
        failures,
        maxPixelDelta,
        maxProbabilityDelta,
        fixtureMaxPixelDelta,
        fixturePrediction,
      };
    })()`,
  );
  check(
    centering.failures.length === 0 && centering.maxPixelDelta <= 0.005 &&
      centering.maxProbabilityDelta <= 0.001 &&
      centering.fixtureMaxPixelDelta <= 1e-9,
    "Off-center normalization failed on the restored model",
    centering,
  );

  async function responsiveSnapshot(width, height, expectedColumns) {
    await devtools.call("Emulation.setDeviceMetricsOverride", {
      width,
      height,
      deviceScaleFactor: 1,
      mobile: false,
    }, sessionId);
    await delay(250);
    const modes = [];
    for (const mode of ["train", "recognize"]) {
      await evaluate(
        devtools,
        sessionId,
        `document.querySelector("#${mode}-tab").click()`,
      );
      await delay(75);
      modes.push(await evaluate(
        devtools,
        sessionId,
        `(() => {
          const control = document.querySelector(".control-panel").getBoundingClientRect();
          const network = document.querySelector(".network-panel").getBoundingClientRect();
          const canvas = document.querySelector("#network-canvas");
          const canvasBounds = canvas.getBoundingClientRect();
          return {
            mode: state.activeTab,
            innerWidth: window.innerWidth,
            clientWidth: document.documentElement.clientWidth,
            documentScrollWidth: document.documentElement.scrollWidth,
            bodyScrollWidth: document.body.scrollWidth,
            control: { left: control.left, right: control.right, top: control.top, bottom: control.bottom },
            network: { left: network.left, right: network.right, top: network.top, bottom: network.bottom },
            canvas: { width: canvasBounds.width, height: canvasBounds.height },
            canvasBacking: { width: canvas.width, height: canvas.height },
            canvasCount: document.querySelectorAll("#network-canvas").length,
            networkVisible: canvasBounds.width > 0 && canvasBounds.height > 0 &&
              getComputedStyle(canvas).display !== "none",
            rendererSame: window.__enenUnifiedIdentity.renderer === networkRenderer &&
              window.__enenUnifiedIdentity.canvas === canvas,
            shape: networkRenderer.snapshot?.layerSizes || [],
          };
        })()`,
      ));
    }
    for (const snapshot of modes) {
      const noOverflow = snapshot.documentScrollWidth <= snapshot.clientWidth + 1 &&
        snapshot.bodyScrollWidth <= snapshot.clientWidth + 1 &&
        snapshot.control.left >= -1 && snapshot.control.right <= snapshot.innerWidth + 1 &&
        snapshot.network.left >= -1 && snapshot.network.right <= snapshot.innerWidth + 1;
      const layoutMatches = expectedColumns
        ? snapshot.network.left >= snapshot.control.right - 1 &&
          Math.abs(snapshot.network.top - snapshot.control.top) <= 2
        : snapshot.network.top >= snapshot.control.bottom - 1;
      check(
        snapshot.mode === (snapshot === modes[0] ? "train" : "recognize") &&
          noOverflow && layoutMatches && snapshot.canvasCount === 1 &&
          snapshot.networkVisible && snapshot.canvasBacking.width > 0 &&
          snapshot.canvasBacking.height > 0 && snapshot.rendererSame &&
          snapshot.shape.join(",") === "64,48,24,10",
        `Responsive ${width}×${height} ${snapshot.mode} layout failed`,
        snapshot,
      );
    }
    return modes;
  }

  const responsive = [];
  responsive.push(...await responsiveSnapshot(1440, 900, true));
  responsive.push(...await responsiveSnapshot(900, 800, true));
  responsive.push(...await responsiveSnapshot(390, 844, false));
  await devtools.call("Emulation.clearDeviceMetricsOverride", {}, sessionId);
  await delay(150);

  const corruption = await evaluate(
    devtools,
    sessionId,
    `(async () => {
      const record = await readStoredModel();
      const validBytes = new Uint8Array(record.model);
      const oversized = new Uint8Array(1_000_001);
      oversized.set(validBytes);
      record.model = oversized.buffer;
      await writeStoredModel(record);
      const written = await readStoredModel();
      return {
        version: written.version,
        revision: written.revision,
        magicPreserved: new Uint8Array(written.model)[0] === validBytes[0],
        length: written.model.byteLength,
      };
    })()`,
  );
  check(
    corruption.version === 2 && corruption.revision === storedModel.revision &&
      corruption.magicPreserved && corruption.length === 1_000_001,
    "Could not create the deterministic oversized IndexedDB fixture",
    corruption,
  );

  await devtools.call("Page.reload", { ignoreCache: true }, sessionId);
  const fallbackReady = await waitFor(
    devtools,
    sessionId,
    readyExpression,
    (value) => value.runtime === "C / Wasm ready" || value.runtime === "Load failed",
    TRAINING_DEADLINE_MS,
    "Corrupt-record fallback",
  );
  check(
    fallbackReady.runtime === "C / Wasm ready" && fallbackReady.origin === expectedOrigin,
    "Oversized record prevented a usable same-origin fallback",
    fallbackReady,
  );
  const fallback = await evaluate(
    devtools,
    sessionId,
    `(async () => {
      const canvas = document.querySelector("#network-canvas");
      const bounds = canvas.getBoundingClientRect();
      return {
        recordDeleted: (await readStoredModel()) === undefined,
        storedModelLoaded: state.storedModelLoaded,
        modelReady: state.modelReady,
        modelTrained: state.modelTrained,
        configured: state.configured,
        training: state.training,
        shape: modelShape(),
        epochsTrained: state.wasm._web_epochs_trained(),
        architecture: document.querySelector("#architecture").value,
        epochs: document.querySelector("#epochs").value,
        learningRate: document.querySelector("#learning-rate").value,
        locked: document.querySelector("#recognition-panel").getAttribute("aria-disabled"),
        drawingDisabled: document.querySelector("#drawing-canvas").getAttribute("aria-disabled"),
        drawToolDisabled: document.querySelector("#draw-tool-button").disabled,
        eraserToolDisabled: document.querySelector("#eraser-tool-button").disabled,
        drawPressed: document.querySelector("#draw-tool-button").getAttribute("aria-pressed"),
        eraserPressed: document.querySelector("#eraser-tool-button").getAttribute("aria-pressed"),
        drawingTool: state.drawingTool,
        canvasTool: document.querySelector("#drawing-canvas").dataset.tool,
        clearDisabled: document.querySelector("#clear-button").disabled,
        exampleDisabled: document.querySelector("#example-button").disabled,
        metrics: [
          document.querySelector("#test-accuracy").textContent,
          document.querySelector("#test-loss").textContent,
          document.querySelector("#drawing-accuracy").textContent,
        ],
        storageStatus: document.querySelector("#storage-status").textContent,
        networkCount: document.querySelectorAll("#network-canvas").length,
        networkVisible: bounds.width > 0 && bounds.height > 0,
        networkShape: networkRenderer.snapshot?.layerSizes || [],
      };
    })()`,
  );
  check(
    fallback.recordDeleted && !fallback.storedModelLoaded && !fallback.modelReady &&
      !fallback.modelTrained && !fallback.configured && !fallback.training &&
      fallback.shape.join(",") === "64,128,64,10" && fallback.epochsTrained === 0 &&
      fallback.architecture === "128,64" && fallback.epochs === "25" &&
      fallback.learningRate === "0.05" && fallback.locked === "true" &&
      fallback.drawingDisabled === "true" && fallback.drawToolDisabled &&
      fallback.eraserToolDisabled && fallback.drawPressed === "true" &&
      fallback.eraserPressed === "false" && fallback.drawingTool === "draw" &&
      fallback.canvasTool === "draw" && fallback.clearDisabled &&
      fallback.exampleDisabled && fallback.metrics.every((value) => value === "—") &&
      fallback.storageStatus.includes("No saved model") && fallback.networkCount === 1 &&
      fallback.networkVisible && fallback.networkShape.join(",") === "64,128,64,10",
    "Oversized IndexedDB record was not revision-safely deleted with a clean locked fallback",
    fallback,
  );

  console.log(
    `browser app ready; 48→24 C frames ${(firstFrameDelay / 1000).toFixed(2)}s/` +
    `${(frameSpacing / 1000).toFixed(2)}s, UCI ${trained.test}, drawing ${trained.drawing}, ` +
    `${storedModel.storedLength} persisted bytes, reload exact, ` +
    `off-center delta ${centering.maxPixelDelta}, responsive ${responsive.length} states, ` +
    "oversized-record fallback clean",
  );
} finally {
  await stopBrowser(browser);
  await new Promise((resolveClose) => server.close(resolveClose));
  await rm(profile, {
    recursive: true,
    force: true,
    maxRetries: 5,
    retryDelay: 100,
  });
}

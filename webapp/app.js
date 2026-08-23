"use strict";

const TRAIN_DATA_URL = "assets/optdigits.tra";
const TEST_DATA_URL = "assets/optdigits.tes";
const WASM_URL = "enen.wasm";
const SOURCE_URL = "assets/nn.c";
const TRAIN_DATA_PATH = "/optdigits.tra";
const TEST_DATA_PATH = "/optdigits.tes";
const MODEL_PATH = "/last-validated.model";
const MODEL_DATABASE = "enen-models";
const MODEL_STORE = "validated-models";
const MODEL_KEY = "last";
const MODEL_RECORD_VERSION = 2;
const MODEL_MAGIC = 0x4e454e45;
const MODEL_FORMAT_VERSION = 1;
const MODEL_LAYER_COUNT = 4;
const MODEL_STORAGE_TIMEOUT_MS = 5000;
const STARTUP_TIMEOUT_MS = 15000;
const DRAWING_MEMORY_PADDING = 24;
const DRAW_STROKE_WIDTH = 22;
const ERASER_STROKE_WIDTH = 34;
const TRAINING_SEED = 5;
const TRAINING_BATCH_SIZE = 128;
const FAST_TRAINING_BATCH_SIZE = 100000;
const VISUALIZATION_INTERVAL_MS = 500;
const ACTIVATION_TRANSITION_MS = 320;
const RECOGNITION_VISUALIZATION_INTERVAL_MS = 150;
const RECOGNITION_TRANSITION_MS = 110;
const SYNTHETIC_TRAINING_PER_DIGIT = 80;
const DRAWING_CHECK_PER_DIGIT = 16;
const MINIMUM_TEST_ACCURACY = 0.94;
const MINIMUM_DRAWING_ACCURACY = 0.8;
const MINIMUM_DRAWING_CLASS_ACCURACY = 0.65;

const ui = {
  brand: document.querySelector(".brand"),
  trainTab: document.querySelector("#train-tab"),
  recognizeTab: document.querySelector("#recognize-tab"),
  trainPanel: document.querySelector("#train-panel"),
  recognitionPanel: document.querySelector("#recognition-panel"),
  sourceButton: document.querySelector("#source-button"),
  sourceDialog: document.querySelector("#source-dialog"),
  sourceCloseButton: document.querySelector("#source-close-button"),
  runtimeStatus: document.querySelector("#runtime-status"),
  setupView: document.querySelector("#setup-view"),
  modelOptions: document.querySelector("#model-options"),
  architecture: document.querySelector("#architecture"),
  architectureNote: document.querySelector("#architecture-note"),
  epochs: document.querySelector("#epochs"),
  epochsValue: document.querySelector("#epochs-value"),
  learningRate: document.querySelector("#learning-rate"),
  newModelButton: document.querySelector("#new-model-button"),
  trainingStatus: document.querySelector("#training-status"),
  trainButton: document.querySelector("#train-button"),
  trainingStage: document.querySelector("#training-stage"),
  trainingStageTitle: document.querySelector("#training-stage-title"),
  networkCanvas: document.querySelector("#network-canvas"),
  networkStatus: document.querySelector("#network-status"),
  networkShape: document.querySelector("#network-shape"),
  networkSampleCanvas: document.querySelector("#network-sample-canvas"),
  networkSampleLabel: document.querySelector("#network-sample-label"),
  networkSampleCaption: document.querySelector("#network-sample-caption"),
  storageStatus: document.querySelector("#storage-status"),
  visualizationStatus: document.querySelector("#visualization-status"),
  layerSummary: document.querySelector("#layer-summary"),
  trainingProgress: document.querySelector("#training-progress"),
  epochLabel: document.querySelector("#epoch-label"),
  testAccuracy: document.querySelector("#test-accuracy"),
  testLoss: document.querySelector("#test-loss"),
  drawingAccuracy: document.querySelector("#drawing-accuracy"),
  modelStatus: document.querySelector("#model-status"),
  recognitionModelStatus: document.querySelector("#recognition-model-status"),
  pauseAnimationButton: document.querySelector("#pause-animation-button"),
  skipAnimationButton: document.querySelector("#skip-animation-button"),
  completionActions: document.querySelector("#completion-actions"),
  configureAgainButton: document.querySelector("#configure-again-button"),
  goRecognizeButton: document.querySelector("#go-recognize-button"),
  trainingAnnouncement: document.querySelector("#training-announcement"),
  drawingCanvas: document.querySelector("#drawing-canvas"),
  canvasMessage: document.querySelector("#canvas-message"),
  drawToolButton: document.querySelector("#draw-tool-button"),
  eraserToolButton: document.querySelector("#eraser-tool-button"),
  clearButton: document.querySelector("#clear-button"),
  exampleButton: document.querySelector("#example-button"),
  predictionDigit: document.querySelector("#prediction-digit"),
  predictionConfidence: document.querySelector("#prediction-confidence"),
  predictionAnnouncement: document.querySelector("#prediction-announcement"),
  probabilities: document.querySelector("#probabilities"),
  sourceCode: document.querySelector("#source-code"),
  sourceStatus: document.querySelector("#source-status"),
  startupFallback: document.querySelector("#startup-fallback"),
};

const state = {
  wasm: null,
  drawingChecks: [],
  readyForTraining: false,
  training: false,
  modelReady: false,
  configured: false,
  animationPaused: false,
  skipAnimation: false,
  reducedMotion: window.matchMedia("(prefers-reduced-motion: reduce)").matches,
  sourceLoaded: false,
  sourceLoading: false,
  storageAvailable: true,
  storedModelLoaded: false,
  activeTab: "train",
  currentConfig: null,
  nextSeed: TRAINING_SEED,
  modelTrained: false,
  drawing: false,
  drawingTool: "draw",
  strokeTool: "draw",
  activePointerId: null,
  lastDrawingPoint: null,
  hasInk: false,
  recognitionFrame: 0,
  networkVisualizationAt: 0,
  networkVisualizationVersion: 0,
  announcePrediction: false,
  exampleIndex: 0,
};

let networkRenderer = null;

const drawingMemoryCanvas = document.createElement("canvas");
let drawingContext = null;
let drawingMemoryContext = null;

class SeededRandom {
  constructor(seed) {
    this.seed = seed >>> 0;
  }

  next() {
    let value = this.seed;
    value ^= value << 13;
    value ^= value >>> 17;
    value ^= value << 5;
    this.seed = value >>> 0;
    return this.seed / 4294967296;
  }
}

function setBadge(element, text, mode = "neutral") {
  element.textContent = text;
  element.classList.toggle("is-ready", mode === "ready");
  element.classList.toggle("is-error", mode === "error");
}

function selectedConfiguration() {
  const [firstHidden, secondHidden] = ui.architecture.value
    .split(",")
    .map(Number);
  return {
    firstHidden,
    secondHidden,
    epochs: Number(ui.epochs.value),
    learningRate: Number(ui.learningRate.value),
  };
}

function parameterCount({ firstHidden, secondHidden }) {
  return (64 + 1) * firstHidden +
    (firstHidden + 1) * secondHidden +
    (secondHidden + 1) * 10;
}

function updateConfigurationCopy() {
  const config = selectedConfiguration();
  ui.epochsValue.value = String(config.epochs);
  ui.epochsValue.textContent = String(config.epochs);
  ui.architectureNote.textContent =
    `${parameterCount(config).toLocaleString()} trainable parameters`;
}

function setSetupControlsDisabled(disabled) {
  ui.architecture.disabled = disabled;
  ui.epochs.disabled = disabled;
  ui.learningRate.disabled = disabled;
  ui.newModelButton.disabled = disabled || !state.readyForTraining;
  ui.trainButton.disabled = disabled || !state.readyForTraining;
}

function switchTab(name, moveFocus = false) {
  const entries = [
    { name: "train", tab: ui.trainTab, panel: ui.trainPanel },
    { name: "recognize", tab: ui.recognizeTab, panel: ui.recognitionPanel },
  ];
  const selected = entries.find((entry) => entry.name === name);
  if (!selected) return;

  for (const entry of entries) {
    const active = entry === selected;
    entry.tab.classList.toggle("is-active", active);
    entry.tab.setAttribute("aria-selected", String(active));
    entry.tab.tabIndex = active ? 0 : -1;
    entry.panel.hidden = !active;
  }
  state.activeTab = name;
  if (moveFocus) selected.tab.focus();
  networkRenderer?.redraw();
}

function handleTabKeydown(event) {
  const tabs = [ui.trainTab, ui.recognizeTab];
  const current = tabs.indexOf(event.currentTarget);
  let next = current;
  if (event.key === "ArrowRight") next = (current + 1) % tabs.length;
  else if (event.key === "ArrowLeft") next = (current - 1 + tabs.length) % tabs.length;
  else if (event.key === "Home") next = 0;
  else if (event.key === "End") next = tabs.length - 1;
  else return;
  event.preventDefault();
  switchTab(tabs[next].dataset.tab, true);
}

function showTrainingStage(show) {
  ui.setupView.hidden = show;
  ui.trainingStage.hidden = !show;
  if (show) {
    void nextPaint().then(() => networkRenderer?.redraw());
  }
}

function resetMetrics() {
  ui.testAccuracy.textContent = "—";
  ui.testLoss.textContent = "—";
  ui.drawingAccuracy.textContent = "—";
}

function setRecognitionEnabled(enabled) {
  state.modelReady = enabled;
  ui.recognitionPanel.classList.toggle("is-disabled", !enabled);
  ui.recognitionPanel.setAttribute("aria-disabled", String(!enabled));
  ui.drawingCanvas.setAttribute("aria-disabled", String(!enabled));
  ui.drawToolButton.disabled = !enabled;
  ui.eraserToolButton.disabled = !enabled;
  ui.clearButton.disabled = !enabled;
  ui.exampleButton.disabled = !enabled;
  updateCanvasToolCopy();
  const text = enabled ? "Model ready" : "Train first";
  const mode = enabled ? "ready" : "neutral";
  setBadge(ui.modelStatus, text, mode);
  setBadge(ui.recognitionModelStatus, text, mode);
  state.networkVisualizationAt = 0;
}

function updateCanvasToolCopy() {
  if (!state.modelReady) {
    ui.canvasMessage.textContent =
      "Recognition unlocks after training passes its checks.";
  } else if (state.drawingTool === "erase") {
    ui.canvasMessage.textContent = state.hasInk
      ? "Drag over the digit to erase ink"
      : "Nothing to erase — select Draw to add ink";
  } else {
    ui.canvasMessage.textContent = "Draw anywhere — centering is automatic";
  }
  ui.drawingCanvas.setAttribute(
    "aria-label",
    state.drawingTool === "erase"
      ? "Erase parts of the handwritten digit"
      : "Draw one digit from zero through nine",
  );
}

function applyDrawingTool(tool = state.drawingTool) {
  const erasing = tool === "erase";
  for (const context of [drawingContext, drawingMemoryContext]) {
    if (!context) continue;
    context.globalCompositeOperation = erasing
      ? "destination-out"
      : "source-over";
    context.lineWidth = erasing ? ERASER_STROKE_WIDTH : DRAW_STROKE_WIDTH;
    context.strokeStyle = "#ffffff";
  }
}

function setDrawingTool(tool) {
  if (tool !== "draw" && tool !== "erase") return;
  state.drawingTool = tool;
  const erasing = tool === "erase";
  ui.drawToolButton.setAttribute("aria-pressed", String(!erasing));
  ui.eraserToolButton.setAttribute("aria-pressed", String(erasing));
  ui.drawingCanvas.classList.toggle("is-erasing", erasing);
  ui.drawingCanvas.dataset.tool = tool;
  applyDrawingTool();
  updateCanvasToolCopy();
}

function setCanvasInkState(hasInk) {
  state.hasInk = hasInk;
  ui.recognitionPanel.classList.toggle("has-ink", hasInk);
  updateCanvasToolCopy();
}

function initializeProbabilityRows() {
  ui.probabilities.innerHTML = Array.from({ length: 10 }, (_, digit) => `
    <div class="probability-row" data-digit="${digit}">
      <span>${digit}</span>
      <span class="probability-track"><span></span></span>
      <span class="probability-value">0%</span>
    </div>
  `).join("");
}

async function fetchBytes(url) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), STARTUP_TIMEOUT_MS);
  try {
    const response = await fetch(url, {
      cache: "no-store",
      signal: controller.signal,
    });
    if (!response.ok) {
      throw new Error(`${url} returned ${response.status}`);
    }
    return new Uint8Array(await response.arrayBuffer());
  } catch (error) {
    if (error.name === "AbortError") {
      throw new Error(`${url} did not respond within 15 seconds`);
    }
    throw error;
  } finally {
    clearTimeout(timeout);
  }
}

function withTimeout(promise, message, milliseconds = STARTUP_TIMEOUT_MS) {
  let timeout;
  const deadline = new Promise((_, reject) => {
    timeout = setTimeout(
      () => reject(new Error(message)),
      milliseconds,
    );
  });
  return Promise.race([promise, deadline]).finally(() => clearTimeout(timeout));
}

function setupInterface() {
  const missing = Object.entries(ui)
    .filter(([name, element]) => name !== "startupFallback" && !element)
    .map(([name]) => name);
  if (missing.length > 0) {
    throw new Error(`The page markup is incomplete (${missing.join(", ")})`);
  }

  drawingContext = ui.drawingCanvas.getContext("2d", {
    willReadFrequently: true,
  });
  drawingMemoryCanvas.width = ui.drawingCanvas.width + DRAWING_MEMORY_PADDING * 2;
  drawingMemoryCanvas.height = ui.drawingCanvas.height + DRAWING_MEMORY_PADDING * 2;
  drawingMemoryContext = drawingMemoryCanvas.getContext("2d", {
    willReadFrequently: true,
  });
  if (!drawingContext || !drawingMemoryContext) {
    throw new Error("This browser could not create the drawing canvas");
  }

  for (const context of [drawingContext, drawingMemoryContext]) {
    context.lineCap = "round";
    context.lineJoin = "round";
    context.lineWidth = DRAW_STROKE_WIDTH;
    context.strokeStyle = "#ffffff";
  }
  setDrawingTool("draw");
}

function layerPositions(layerSizes, width, height) {
  const left = 34;
  const right = 48;
  const top = 28;
  const bottom = 20;
  return layerSizes.map((size, layer) => {
    const x = layerSizes.length === 1
      ? width / 2
      : left + (width - left - right) * layer / (layerSizes.length - 1);
    const span = Math.max(1, height - top - bottom);
    return Array.from({ length: size }, (_, node) => ({
      x,
      y: top + span * (node + 0.5) / size,
    }));
  });
}

class DenseNetworkRenderer {
  constructor(canvas, { transitionMs = ACTIVATION_TRANSITION_MS } = {}) {
    this.canvas = canvas;
    this.transitionMs = transitionMs;
    this.context = canvas.getContext("2d");
    if (!this.context) {
      throw new Error("This browser could not create the network canvas");
    }
    this.connectionCanvas = document.createElement("canvas");
    this.snapshot = null;
    this.previousActivations = null;
    this.frame = 0;
    this.width = 0;
    this.height = 0;
    this.dpr = 1;
    this.renderCount = 0;
    this.resizeObserver = typeof ResizeObserver === "function"
      ? new ResizeObserver(() => this.redraw())
      : null;
    this.resizeObserver?.observe(canvas);
  }

  ensureSize() {
    const bounds = this.canvas.getBoundingClientRect();
    const width = bounds.width > 0 ? Math.round(bounds.width) : 960;
    const height = bounds.height > 0 ? Math.round(bounds.height) : 440;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const backingWidth = Math.round(width * dpr);
    const backingHeight = Math.round(height * dpr);
    const changed = this.canvas.width !== backingWidth ||
      this.canvas.height !== backingHeight;
    if (changed) {
      this.canvas.width = backingWidth;
      this.canvas.height = backingHeight;
      this.connectionCanvas.width = backingWidth;
      this.connectionCanvas.height = backingHeight;
    }
    this.width = width;
    this.height = height;
    this.dpr = dpr;
    return changed;
  }

  buildConnections() {
    if (!this.snapshot) return;
    const context = this.connectionCanvas.getContext("2d");
    const { layerSizes, weights, activations } = this.snapshot;
    const positions = layerPositions(layerSizes, this.width, this.height);
    context.setTransform(1, 0, 0, 1, 0, 0);
    context.clearRect(0, 0, this.connectionCanvas.width, this.connectionCanvas.height);
    context.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    context.lineWidth = 0.55;

    let activationOffset = 0;
    for (let layer = 0; layer < layerSizes.length - 1; layer += 1) {
      const currentSize = layerSizes[layer];
      const nextSize = layerSizes[layer + 1];
      const matrix = weights[layer];
      const nextActivationOffset = activationOffset + currentSize;
      let maximum = 0;
      for (const weight of matrix) maximum = Math.max(maximum, Math.abs(weight));
      maximum ||= 1;

      const paths = Array.from({ length: 8 }, () => new Path2D());
      const activePositive = new Path2D();
      const activeNegative = new Path2D();
      for (let input = 0; input < currentSize; input += 1) {
        const from = positions[layer][input];
        const sourceActivation = activations[activationOffset + input];
        for (let output = 0; output < nextSize; output += 1) {
          const to = positions[layer + 1][output];
          const weight = matrix[input * nextSize + output];
          const relative = Math.min(0.999, Math.abs(weight) / maximum);
          const bucket = Math.min(3, Math.floor(relative * 4));
          const sign = weight < 0 ? 1 : 0;
          const path = paths[sign * 4 + bucket];
          path.moveTo(from.x, from.y);
          path.lineTo(to.x, to.y);

          const targetActivation = activations[nextActivationOffset + output];
          const activity = Math.sqrt(sourceActivation * targetActivation) * relative;
          if (activity > 0.38) {
            const activePath = weight < 0 ? activeNegative : activePositive;
            activePath.moveTo(from.x, from.y);
            activePath.lineTo(to.x, to.y);
          }
        }
      }
      for (let bucket = 0; bucket < 4; bucket += 1) {
        const alpha = 0.014 + bucket * 0.012;
        context.strokeStyle = `rgba(35, 79, 108, ${alpha})`;
        context.stroke(paths[bucket]);
        context.strokeStyle = `rgba(143, 66, 52, ${alpha})`;
        context.stroke(paths[4 + bucket]);
      }
      context.lineWidth = 0.8;
      context.strokeStyle = "rgba(35, 79, 108, 0.13)";
      context.stroke(activePositive);
      context.strokeStyle = "rgba(143, 66, 52, 0.12)";
      context.stroke(activeNegative);
      activationOffset = nextActivationOffset;
    }
  }

  draw(progress = 1) {
    if (!this.snapshot || !this.context) return;
    const { layerSizes, activations, label } = this.snapshot;
    const previous = this.previousActivations?.length === activations.length
      ? this.previousActivations
      : activations;
    const positions = layerPositions(layerSizes, this.width, this.height);
    const context = this.context;
    context.setTransform(1, 0, 0, 1, 0, 0);
    context.fillStyle = "#f3f3ef";
    context.fillRect(0, 0, this.canvas.width, this.canvas.height);
    context.drawImage(this.connectionCanvas, 0, 0);
    context.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);

    context.font = "700 9px ui-monospace, SFMono-Regular, monospace";
    context.textAlign = "center";
    context.fillStyle = "#555b5e";
    const layerNames = ["INPUT", "HIDDEN 1", "HIDDEN 2", "OUTPUT"];
    for (let layer = 0; layer < layerSizes.length; layer += 1) {
      context.fillText(
        `${layerNames[layer] || `LAYER ${layer}`} · ${layerSizes[layer]}`,
        positions[layer][0].x,
        12,
      );
    }

    const eased = 1 - (1 - progress) ** 3;
    let offset = 0;
    for (let layer = 0; layer < layerSizes.length; layer += 1) {
      const size = layerSizes[layer];
      const spacing = (this.height - 48) / size;
      const baseRadius = Math.max(0.55, Math.min(2.6, spacing * 0.3));
      for (let node = 0; node < size; node += 1) {
        const index = offset + node;
        const activation = previous[index] +
          (activations[index] - previous[index]) * eased;
        const { x, y } = positions[layer][node];
        const radius = baseRadius + Math.min(1.9, activation * 1.75);

        context.beginPath();
        context.fillStyle = `rgba(22, 58, 82, ${0.14 + activation * 0.86})`;
        context.arc(x, y, radius, 0, Math.PI * 2);
        context.fill();
      }
      offset += size;
    }

    const outputLayer = layerSizes.length - 1;
    const outputOffset = activations.length - layerSizes[outputLayer];
    let predicted = 0;
    for (let digit = 1; digit < layerSizes[outputLayer]; digit += 1) {
      if (activations[outputOffset + digit] > activations[outputOffset + predicted]) {
        predicted = digit;
      }
    }
    context.font = "700 9px ui-monospace, SFMono-Regular, monospace";
    context.textAlign = "left";
    for (let digit = 0; digit < layerSizes[outputLayer]; digit += 1) {
      const point = positions[outputLayer][digit];
      context.fillStyle = digit === predicted ? "#111820" : "#6a6f72";
      context.fillText(String(digit), point.x + 8, point.y + 3);
      if (digit === label || digit === predicted) {
        context.beginPath();
        context.lineWidth = digit === predicted ? 1.6 : 1;
        context.strokeStyle = digit === predicted ? "#173f5b" : "#8f4234";
        context.arc(point.x, point.y, 5.2, 0, Math.PI * 2);
        context.stroke();
      }
    }
  }

  render(snapshot, animate = true) {
    cancelAnimationFrame(this.frame);
    this.previousActivations = this.snapshot?.activations || snapshot.activations;
    this.snapshot = snapshot;
    this.ensureSize();
    this.buildConnections();
    this.renderCount += 1;
    this.canvas.dataset.renderCount = String(this.renderCount);
    this.canvas.dataset.activationVersion = String(snapshot.version);

    if (!animate || state.reducedMotion) {
      this.draw(1);
      this.previousActivations = snapshot.activations;
      return;
    }
    const started = performance.now();
    const step = (now) => {
      const progress = Math.min(1, (now - started) / this.transitionMs);
      this.draw(progress);
      if (progress < 1) this.frame = requestAnimationFrame(step);
      else this.previousActivations = snapshot.activations;
    };
    this.frame = requestAnimationFrame(step);
  }

  redraw() {
    if (!this.snapshot) return;
    const changed = this.ensureSize();
    if (changed) this.buildConnections();
    this.draw(1);
  }

  freeze() {
    cancelAnimationFrame(this.frame);
    this.draw(1);
  }

  clear() {
    cancelAnimationFrame(this.frame);
    this.snapshot = null;
    this.previousActivations = null;
    this.renderCount = 0;
    this.ensureSize();
    this.context.setTransform(1, 0, 0, 1, 0, 0);
    this.context.fillStyle = "#f3f3ef";
    this.context.fillRect(0, 0, this.canvas.width, this.canvas.height);
    this.canvas.dataset.renderCount = "0";
    delete this.canvas.dataset.activationVersion;
  }
}

function drawNetworkSample(pixels) {
  const canvas = ui.networkSampleCanvas;
  const context = canvas.getContext("2d");
  if (!context) throw new Error("This browser could not create the sample canvas");
  const cellWidth = canvas.width / 8;
  const cellHeight = canvas.height / 8;
  context.fillStyle = "#f4f4f0";
  context.fillRect(0, 0, canvas.width, canvas.height);
  for (let row = 0; row < 8; row += 1) {
    for (let column = 0; column < 8; column += 1) {
      const value = Math.max(0, Math.min(1, pixels[row * 8 + column]));
      context.fillStyle = `rgba(24, 55, 78, ${0.04 + value * 0.96})`;
      context.fillRect(
        column * cellWidth + 1,
        row * cellHeight + 1,
        Math.max(1, cellWidth - 2),
        Math.max(1, cellHeight - 2),
      );
    }
  }
}

function readNetworkSnapshot(initial = false) {
  const wasm = state.wasm;
  const layerCount = wasm._web_num_layers();
  const layerSizes = Array.from(
    { length: layerCount },
    (_, layer) => wasm._web_layer_size(layer),
  );
  if (layerCount !== 4 || layerSizes.some((size) => size <= 0)) {
    throw new Error("C runtime reported an invalid network architecture");
  }
  const activationCount = layerSizes.reduce((total, size) => total + size, 0);
  let activations;
  let label;
  if (initial) {
    label = wasm._web_copy_training_sample(0);
    if (label < 0 || wasm._web_inspect_input() < 0) {
      throw new Error("C runtime could not inspect the initial training example");
    }
    activations = new Float64Array(activationCount);
    let offset = 0;
    for (let layer = 0; layer < layerCount; layer += 1) {
      for (let node = 0; node < layerSizes[layer]; node += 1) {
        activations[offset++] = wasm._web_activation(layer, node);
      }
    }
  } else {
    const pointer = wasm._web_activation_snapshot();
    if (!pointer || wasm._web_activation_count() !== activationCount) {
      throw new Error("C runtime did not provide a complete activation snapshot");
    }
    const offset = pointer / Float64Array.BYTES_PER_ELEMENT;
    activations = Float64Array.from(
      wasm.HEAPF64.subarray(offset, offset + activationCount),
    );
    label = wasm._web_last_training_label();
  }

  const weights = [];
  for (let layer = 0; layer < layerCount - 1; layer += 1) {
    const pointer = wasm._web_layer_weights(layer);
    const count = layerSizes[layer] * layerSizes[layer + 1];
    if (!pointer) throw new Error("C runtime did not provide network weights");
    const offset = pointer / Float64Array.BYTES_PER_ELEMENT;
    weights.push(Float64Array.from(wasm.HEAPF64.subarray(offset, offset + count)));
  }
  if (label < 0 || label > 9 ||
      !activations.every((value) => Number.isFinite(value) && value >= 0 && value <= 1) ||
      weights.some((matrix) => !matrix.every(Number.isFinite))) {
    throw new Error("C runtime produced a non-finite visualization state");
  }
  return {
    layerSizes,
    activations,
    weights,
    label,
    version: initial ? 0 : wasm._web_activation_version(),
  };
}

function renderTrainingSnapshot(initial = false, animate = true) {
  const snapshot = readNetworkSnapshot(initial);
  drawNetworkSample(snapshot.activations.subarray(0, 64));
  ui.networkSampleLabel.textContent = String(snapshot.label);
  ui.networkSampleCaption.textContent = initial
    ? "Before the first weight update"
    : `Activation snapshot ${snapshot.version.toLocaleString()}`;
  ui.networkShape.textContent = snapshot.layerSizes.join(" → ");
  ui.networkStatus.textContent = initial
    ? "Training · initial weights"
    : `Training · sample ${snapshot.label}`;
  networkRenderer.transitionMs = ACTIVATION_TRANSITION_MS;
  networkRenderer.render(snapshot, animate);
  return snapshot;
}

function readLiveNetworkSnapshot() {
  const wasm = state.wasm;
  const layerCount = wasm._web_num_layers();
  const layerSizes = Array.from(
    { length: layerCount },
    (_, layer) => wasm._web_layer_size(layer),
  );
  if (layerCount !== 4 || layerSizes.some((size) => size <= 0)) {
    throw new Error("C runtime reported an invalid recognition architecture");
  }

  const activationCount = layerSizes.reduce((total, size) => total + size, 0);
  const activations = new Float64Array(activationCount);
  let activationOffset = 0;
  for (let layer = 0; layer < layerCount; layer += 1) {
    for (let node = 0; node < layerSizes[layer]; node += 1) {
      activations[activationOffset++] = wasm._web_activation(layer, node);
    }
  }

  const weights = [];
  for (let layer = 0; layer < layerCount - 1; layer += 1) {
    const pointer = wasm._web_layer_weights(layer);
    const count = layerSizes[layer] * layerSizes[layer + 1];
    if (!pointer) throw new Error("C runtime did not provide recognition weights");
    const offset = pointer / Float64Array.BYTES_PER_ELEMENT;
    weights.push(Float64Array.from(wasm.HEAPF64.subarray(offset, offset + count)));
  }

  if (!activations.every(
    (value) => Number.isFinite(value) && value >= 0 && value <= 1,
  ) || weights.some((matrix) => !matrix.every(Number.isFinite))) {
    throw new Error("C runtime produced an invalid recognition visualization");
  }
  state.networkVisualizationVersion += 1;
  return {
    layerSizes,
    activations,
    weights,
    label: null,
    version: state.networkVisualizationVersion,
  };
}

function renderRecognitionSnapshot(pixels) {
  // web_predict() averages several nearby variants for stability. For a
  // readable activation diagram, inspect the centered drawing itself once and
  // copy that one truthful C forward pass into the renderer.
  writeWasmInput(pixels);
  const directPrediction = state.wasm._web_inspect_input();
  if (directPrediction < 0) {
    throw new Error("C runtime could not inspect the centered drawing");
  }
  const snapshot = readLiveNetworkSnapshot();
  drawNetworkSample(pixels);
  ui.networkSampleLabel.textContent = String(directPrediction);
  ui.networkSampleCaption.textContent = "Centered recognition input";
  ui.networkShape.textContent = snapshot.layerSizes.join(" → ");
  ui.networkStatus.textContent =
    `Recognizing · strongest direct output ${directPrediction}`;
  networkRenderer.transitionMs = RECOGNITION_TRANSITION_MS;
  networkRenderer.render(snapshot, true);
  state.networkVisualizationAt = performance.now();
  return snapshot;
}

function renderNetworkAtRest(status = "Weights · no input") {
  if (!state.wasm || !networkRenderer) return null;
  const pixels = new Float64Array(64);
  writeWasmInput(pixels);
  if (state.wasm._web_inspect_input() < 0) {
    throw new Error("C runtime could not inspect the resting model");
  }
  const snapshot = readLiveNetworkSnapshot();
  drawNetworkSample(pixels);
  ui.networkSampleLabel.textContent = "—";
  ui.networkSampleCaption.textContent = "No input";
  ui.networkShape.textContent = snapshot.layerSizes.join(" → ");
  ui.networkStatus.textContent = status;
  networkRenderer.transitionMs = ACTIVATION_TRANSITION_MS;
  networkRenderer.render(snapshot, false);
  state.networkVisualizationAt = 0;
  return snapshot;
}

function showStartupError(error) {
  console.error(error);
  const message = window.location.protocol === "file:"
    ? "Open this page through the local web server, not as a file."
    : `Could not start the C / WebAssembly app: ${error.message}`;
  if (ui.runtimeStatus) {
    setBadge(ui.runtimeStatus, "Load failed", "error");
  }
  if (ui.trainingStatus) {
    ui.trainingStatus.textContent = message;
  } else if (ui.startupFallback) {
    ui.startupFallback.hidden = false;
    ui.startupFallback.textContent = message;
  }
}

function writeWasmInput(pixels) {
  // Emscripten can replace its ArrayBuffer when memory grows. Reading HEAPF64
  // for every copy ensures this view always targets the current Wasm memory.
  const offset = state.wasm._web_input_buffer() / Float64Array.BYTES_PER_ELEMENT;
  state.wasm.HEAPF64.set(pixels, offset);
}

function readWasmInput() {
  const offset = state.wasm._web_input_buffer() / Float64Array.BYTES_PER_ELEMENT;
  return Float64Array.from(state.wasm.HEAPF64.subarray(offset, offset + 64));
}

function canvasToPixels(canvas) {
  const context = canvas.getContext("2d", { willReadFrequently: true });
  const width = canvas.width;
  const height = canvas.height;
  const image = context.getImageData(0, 0, width, height);
  let left = width;
  let right = -1;
  let top = height;
  let bottom = -1;

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      if (image.data[(y * width + x) * 4 + 3] > 10) {
        if (x < left) left = x;
        if (x > right) right = x;
        if (y < top) top = y;
        if (y > bottom) bottom = y;
      }
    }
  }
  if (right < left || bottom < top) {
    return null;
  }

  // Crop the full in-memory ink bounds and place them at the center of the
  // normalization raster. Position on the drawing surface therefore cannot
  // become a position offset in the 8x8 model input.
  const sourceWidth = right - left + 1;
  const sourceHeight = bottom - top + 1;
  const scale = Math.min(64 / sourceWidth, 64 / sourceHeight);
  const targetWidth = sourceWidth * scale;
  const targetHeight = sourceHeight * scale;

  const targetLeft = (80 - targetWidth) / 2;
  const targetTop = (80 - targetHeight) / 2;
  const normalized = new Float64Array(80 * 80);
  for (let targetY = 0; targetY < 80; targetY += 1) {
    const relativeY = (targetY + 0.5 - targetTop) / targetHeight;
    if (relativeY < 0 || relativeY >= 1) continue;
    const sourceY = top + relativeY * sourceHeight - 0.5;
    for (let targetX = 0; targetX < 80; targetX += 1) {
      const relativeX = (targetX + 0.5 - targetLeft) / targetWidth;
      if (relativeX < 0 || relativeX >= 1) continue;
      const sourceX = left + relativeX * sourceWidth - 0.5;
      normalized[targetY * 80 + targetX] = sampleAlpha(
        image.data,
        width,
        height,
        sourceX,
        sourceY,
      );
    }
  }

  const pixels = new Float64Array(64);
  for (let row = 0; row < 8; row += 1) {
    for (let column = 0; column < 8; column += 1) {
      let alpha = 0;
      for (let y = 0; y < 10; y += 1) {
        for (let x = 0; x < 10; x += 1) {
          const pixel = (row * 10 + y) * 80 + column * 10 + x;
          alpha += normalized[pixel];
        }
      }
      pixels[row * 8 + column] = Math.min(1, alpha / 74);
    }
  }
  return pixels;
}

function sampleAlpha(rgba, width, height, x, y) {
  const left = Math.floor(x);
  const top = Math.floor(y);
  const fractionX = x - left;
  const fractionY = y - top;
  let alpha = 0;
  for (let yStep = 0; yStep <= 1; yStep += 1) {
    const sourceY = top + yStep;
    if (sourceY < 0 || sourceY >= height) continue;
    const weightY = yStep === 0 ? 1 - fractionY : fractionY;
    for (let xStep = 0; xStep <= 1; xStep += 1) {
      const sourceX = left + xStep;
      if (sourceX < 0 || sourceX >= width) continue;
      const weightX = xStep === 0 ? 1 - fractionX : fractionX;
      alpha += rgba[(sourceY * width + sourceX) * 4 + 3] / 255 *
        weightX * weightY;
    }
  }
  return alpha;
}

function drawVectorDigit(context, digit) {
  context.beginPath();
  switch (digit) {
    case 0:
      context.ellipse(0, 0, 20, 30, 0, 0, Math.PI * 2);
      break;
    case 1:
      context.moveTo(-12, -20);
      context.lineTo(0, -30);
      context.lineTo(0, 29);
      context.moveTo(-13, 29);
      context.lineTo(14, 29);
      break;
    case 2:
      context.moveTo(-20, -20);
      context.bezierCurveTo(-8, -35, 21, -33, 22, -14);
      context.bezierCurveTo(23, -2, 6, 9, -20, 29);
      context.lineTo(23, 29);
      break;
    case 3:
      context.moveTo(-20, -24);
      context.bezierCurveTo(-3, -34, 22, -31, 22, -12);
      context.bezierCurveTo(22, -2, 9, 2, -5, 3);
      context.bezierCurveTo(9, 3, 24, 9, 23, 21);
      context.bezierCurveTo(21, 34, -5, 35, -21, 26);
      break;
    case 4:
      context.moveTo(15, 30);
      context.lineTo(15, -30);
      context.moveTo(15, -26);
      context.lineTo(-22, 12);
      context.lineTo(25, 12);
      break;
    case 5:
      context.moveTo(20, -29);
      context.lineTo(-18, -29);
      context.lineTo(-20, -3);
      context.bezierCurveTo(-3, -9, 21, -7, 22, 13);
      context.bezierCurveTo(23, 34, -7, 36, -22, 25);
      break;
    case 6:
      context.moveTo(17, -27);
      context.bezierCurveTo(-10, -34, -23, -5, -20, 17);
      context.bezierCurveTo(-18, 36, 18, 37, 22, 15);
      context.bezierCurveTo(24, -3, -8, -7, -19, 8);
      break;
    case 7:
      context.moveTo(-22, -29);
      context.lineTo(23, -29);
      context.lineTo(-4, 31);
      break;
    case 8:
      context.ellipse(0, -15, 17, 16, 0, 0, Math.PI * 2);
      context.moveTo(21, 15);
      context.ellipse(0, 15, 21, 18, 0, 0, Math.PI * 2);
      break;
    case 9:
      context.moveTo(18, -5);
      context.bezierCurveTo(7, 8, -20, 4, -20, -15);
      context.bezierCurveTo(-20, -36, 18, -36, 20, -15);
      context.bezierCurveTo(22, 2, 17, 20, -3, 31);
      break;
    default:
      return;
  }
  context.stroke();
}

function drawAlternateDigit(context, digit) {
  context.beginPath();
  switch (digit) {
    case 0:
      context.ellipse(140, 140, 62, 96, 0, 0, Math.PI * 2);
      break;
    case 1:
      context.moveTo(115, 70);
      context.lineTo(142, 46);
      context.lineTo(142, 232);
      break;
    case 2:
      context.moveTo(76, 88);
      context.bezierCurveTo(104, 35, 205, 47, 202, 98);
      context.bezierCurveTo(200, 128, 145, 163, 77, 224);
      context.lineTo(210, 224);
      break;
    case 3:
      context.moveTo(82, 68);
      context.bezierCurveTo(185, 30, 215, 88, 144, 132);
      context.bezierCurveTo(225, 132, 211, 238, 77, 216);
      break;
    case 4:
      context.moveTo(192, 235);
      context.lineTo(192, 45);
      context.moveTo(190, 55);
      context.lineTo(72, 174);
      context.lineTo(225, 174);
      break;
    case 5:
      context.moveTo(205, 56);
      context.lineTo(85, 56);
      context.lineTo(78, 132);
      context.bezierCurveTo(180, 102, 226, 151, 202, 209);
      context.bezierCurveTo(175, 264, 92, 238, 70, 211);
      break;
    case 6:
      context.moveTo(194, 60);
      context.bezierCurveTo(91, 35, 62, 145, 81, 207);
      context.bezierCurveTo(101, 270, 218, 235, 203, 165);
      context.bezierCurveTo(189, 105, 94, 113, 79, 175);
      break;
    case 7:
      context.moveTo(66, 57);
      context.lineTo(215, 57);
      context.lineTo(112, 235);
      break;
    case 8:
      context.ellipse(140, 93, 50, 46, 0, 0, Math.PI * 2);
      context.moveTo(204, 181);
      context.ellipse(140, 181, 63, 55, 0, 0, Math.PI * 2);
      break;
    case 9:
      context.moveTo(202, 147);
      context.bezierCurveTo(170, 180, 76, 154, 78, 92);
      context.bezierCurveTo(80, 29, 198, 32, 204, 96);
      context.bezierCurveTo(210, 158, 181, 211, 104, 235);
      break;
    default:
      return;
  }
  context.stroke();
}

function renderSyntheticDigit(canvas, label, random, useVector) {
  const context = canvas.getContext("2d", { willReadFrequently: true });
  const size = Math.min(canvas.width, canvas.height);
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.save();
  context.translate(
    canvas.width / 2 + (random.next() - 0.5) * size * 0.05,
    canvas.height / 2 + (random.next() - 0.5) * size * 0.04,
  );
  context.rotate((random.next() - 0.5) * 0.2);
  context.scale(0.9 + random.next() * 0.2, 0.9 + random.next() * 0.18);

  if (useVector) {
    const unit = size / 80;
    context.scale(unit, unit);
    context.strokeStyle = "#ffffff";
    context.lineWidth = 5.5 + random.next() * 2.2;
    context.lineCap = "round";
    context.lineJoin = "round";
    drawVectorDigit(context, label);
  } else {
    const fonts = ["Arial", "Verdana", "Georgia", "Times New Roman"];
    const font = fonts[Math.floor(random.next() * fonts.length)];
    const weight = random.next() < 0.45 ? "600" : "400";
    context.fillStyle = "#ffffff";
    context.font = `${weight} ${size * (0.68 + random.next() * 0.08)}px ${font}`;
    context.textAlign = "center";
    context.textBaseline = "middle";
    context.fillText(String(label), 0, size * 0.015);
  }
  context.restore();
}

function renderDrawingCheckDigit(canvas, label, random, useAlternatePath) {
  if (!useAlternatePath) {
    renderSyntheticDigit(canvas, label, random, false);
    return;
  }

  const context = canvas.getContext("2d", { willReadFrequently: true });
  const size = Math.min(canvas.width, canvas.height);
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.save();
  context.translate(
    canvas.width / 2 + (random.next() - 0.5) * size * 0.04,
    canvas.height / 2 + (random.next() - 0.5) * size * 0.035,
  );
  context.rotate((random.next() - 0.5) * 0.15);
  context.scale(0.92 + random.next() * 0.16, 0.92 + random.next() * 0.15);
  context.scale(size / 280, size / 280);
  context.translate(-140, -140);
  context.strokeStyle = "#ffffff";
  context.lineWidth = 18 + random.next() * 5;
  context.lineCap = "round";
  context.lineJoin = "round";
  drawAlternateDigit(context, label);
  context.restore();
}

function createSyntheticSamples(samplesPerDigit, seed) {
  const canvas = document.createElement("canvas");
  canvas.width = 96;
  canvas.height = 96;
  const random = new SeededRandom(seed);
  const samples = [];

  for (let label = 0; label < 10; label += 1) {
    for (let index = 0; index < samplesPerDigit; index += 1) {
      renderSyntheticDigit(canvas, label, random, index % 2 === 0);
      const pixels = canvasToPixels(canvas);
      if (!pixels) {
        throw new Error(`Could not rasterize synthetic digit ${label}`);
      }
      samples.push({ label, pixels });
    }
  }
  return samples;
}

function createDrawingCheckSamples(samplesPerDigit, seed) {
  const canvas = document.createElement("canvas");
  canvas.width = 96;
  canvas.height = 96;
  const random = new SeededRandom(seed);
  const samples = [];

  for (let label = 0; label < 10; label += 1) {
    for (let index = 0; index < samplesPerDigit; index += 1) {
      renderDrawingCheckDigit(canvas, label, random, index % 2 === 0);
      const pixels = canvasToPixels(canvas);
      if (!pixels) {
        throw new Error(`Could not rasterize drawing check ${label}`);
      }
      samples.push({ label, pixels });
    }
  }
  return samples;
}

async function prepareDrawingData() {
  ui.trainingStatus.textContent = "Preparing normalized handwriting variations…";
  await nextPaint();
  const trainingSamples = createSyntheticSamples(
    SYNTHETIC_TRAINING_PER_DIGIT,
    0x31415926,
  );
  state.drawingChecks = createDrawingCheckSamples(
    DRAWING_CHECK_PER_DIGIT,
    0x27182818,
  );

  state.wasm._web_clear_synthetic_samples();
  for (let index = 0; index < trainingSamples.length; index += 1) {
    const sample = trainingSamples[index];
    writeWasmInput(sample.pixels);
    if (state.wasm._web_add_synthetic_sample(sample.label) !== 0) {
      throw new Error("C runtime could not store a handwriting sample");
    }
    if (index > 0 && index % 100 === 0) {
      ui.trainingStatus.textContent =
        `Preparing normalized handwriting variations… ${index} / ${trainingSamples.length}`;
      await nextPaint();
    }
  }
}

function openModelDatabase() {
  return new Promise((resolve, reject) => {
    if (!window.indexedDB) {
      reject(new Error("IndexedDB is unavailable"));
      return;
    }
    const request = window.indexedDB.open(MODEL_DATABASE, 1);
    let settled = false;
    const timeout = setTimeout(() => {
      if (settled) return;
      settled = true;
      try {
        request.transaction?.abort();
      } catch {
        // The open or upgrade request may already have ended.
      }
      reject(new Error("Browser model storage did not respond within 5 seconds"));
    }, MODEL_STORAGE_TIMEOUT_MS);
    request.onupgradeneeded = () => {
      const database = request.result;
      if (!database.objectStoreNames.contains(MODEL_STORE)) {
        database.createObjectStore(MODEL_STORE);
      }
    };
    request.onsuccess = () => {
      if (settled) {
        request.result.close();
        return;
      }
      settled = true;
      clearTimeout(timeout);
      resolve(request.result);
    };
    request.onerror = () => {
      if (settled) return;
      settled = true;
      clearTimeout(timeout);
      reject(request.error || new Error("Could not open model storage"));
    };
    request.onblocked = () => {
      if (settled) return;
      settled = true;
      clearTimeout(timeout);
      reject(new Error("Browser model storage is open in an incompatible tab"));
    };
  });
}

async function modelStorageOperation(mode, action) {
  const database = await openModelDatabase();
  try {
    return await new Promise((resolve, reject) => {
      const transaction = database.transaction(MODEL_STORE, mode);
      const store = transaction.objectStore(MODEL_STORE);
      let request;
      let settled = false;
      const timeout = setTimeout(() => {
        if (settled) return;
        settled = true;
        try {
          transaction.abort();
        } catch {
          // Completion won the race; its handler will settle the operation.
        }
        reject(new Error("Browser model storage did not respond within 5 seconds"));
      }, MODEL_STORAGE_TIMEOUT_MS);
      const fail = (error) => {
        if (settled) return;
        settled = true;
        clearTimeout(timeout);
        reject(error);
      };
      try {
        request = action(store);
      } catch (error) {
        try {
          transaction.abort();
        } catch {
          // The transaction may already have ended after a synchronous error.
        }
        fail(error);
        return;
      }
      transaction.oncomplete = () => {
        if (settled) return;
        settled = true;
        clearTimeout(timeout);
        resolve(request?.result);
      };
      transaction.onerror = () => fail(
        transaction.error || request?.error || new Error("Model storage failed"),
      );
      transaction.onabort = () => fail(
        transaction.error || new Error("Model storage was aborted"),
      );
    });
  } finally {
    database.close();
  }
}

function readStoredModel() {
  return modelStorageOperation("readonly", (store) => store.get(MODEL_KEY));
}

function writeStoredModel(record) {
  return modelStorageOperation("readwrite", (store) => store.put(record, MODEL_KEY));
}

async function deleteStoredModelRevision(revision) {
  const database = await openModelDatabase();
  try {
    return await new Promise((resolve, reject) => {
      const transaction = database.transaction(MODEL_STORE, "readwrite");
      const store = transaction.objectStore(MODEL_STORE);
      let removed = false;
      let settled = false;
      const timeout = setTimeout(() => {
        if (settled) return;
        settled = true;
        try {
          transaction.abort();
        } catch {
          // Completion won the race; its handler will settle the operation.
        }
        reject(new Error("Browser model storage did not respond within 5 seconds"));
      }, MODEL_STORAGE_TIMEOUT_MS);
      const request = store.get(MODEL_KEY);
      request.onsuccess = () => {
        if (request.result?.revision === revision) {
          store.delete(MODEL_KEY);
          removed = true;
        }
      };
      transaction.oncomplete = () => {
        if (settled) return;
        settled = true;
        clearTimeout(timeout);
        resolve(removed);
      };
      const fail = (error) => {
        if (settled) return;
        settled = true;
        clearTimeout(timeout);
        reject(error);
      };
      transaction.onerror = () => fail(
        transaction.error || new Error("Model storage failed"),
      );
      transaction.onabort = () => fail(
        transaction.error || new Error("Model storage was aborted"),
      );
    });
  } finally {
    database.close();
  }
}

function removeVirtualModelFile() {
  try {
    state.wasm.FS.unlink(MODEL_PATH);
  } catch {
    // The in-memory file is optional and may not have been created yet.
  }
}

function modelShape() {
  return Array.from(
    { length: state.wasm._web_num_layers() },
    (_, layer) => state.wasm._web_layer_size(layer),
  );
}

function qualityPassed(testAccuracy, drawingMetrics) {
  return testAccuracy >= MINIMUM_TEST_ACCURACY &&
    drawingMetrics.accuracy >= MINIMUM_DRAWING_ACCURACY &&
    drawingMetrics.minimumClassAccuracy >= MINIMUM_DRAWING_CLASS_ACCURACY;
}

async function persistValidatedModel(metrics) {
  try {
    removeVirtualModelFile();
    const saved = state.wasm.ccall(
      "web_save_model",
      "number",
      ["string"],
      [MODEL_PATH],
    );
    if (saved !== 0) throw new Error("C could not serialize the trained model");
    const bytes = state.wasm.FS.readFile(MODEL_PATH);
    const model = bytes.buffer.slice(
      bytes.byteOffset,
      bytes.byteOffset + bytes.byteLength,
    );
    await writeStoredModel({
      version: MODEL_RECORD_VERSION,
      revision: typeof crypto.randomUUID === "function"
        ? crypto.randomUUID()
        : `${Date.now()}-${Math.random().toString(16).slice(2)}`,
      savedAt: Date.now(),
      model,
      config: { ...state.currentConfig },
      metrics,
    });
    state.storageAvailable = true;
    state.storedModelLoaded = true;
    ui.storageStatus.textContent =
      "Saved in this browser profile for this site";
    return true;
  } catch (error) {
    console.warn("Could not persist the validated model", error);
    state.storageAvailable = false;
    ui.storageStatus.textContent =
      "Current tab only · browser storage unavailable";
    return false;
  } finally {
    removeVirtualModelFile();
  }
}

function serializedModelShape(model) {
  if (!(model instanceof ArrayBuffer) || model.byteLength < 28 ||
      model.byteLength > 1_000_000) {
    return null;
  }
  const view = new DataView(model);
  if (view.getUint32(0, true) !== MODEL_MAGIC ||
      view.getUint32(4, true) !== MODEL_FORMAT_VERSION ||
      view.getUint32(8, true) !== MODEL_LAYER_COUNT) {
    return null;
  }
  const shape = Array.from(
    { length: MODEL_LAYER_COUNT },
    (_, layer) => view.getInt32(12 + layer * 4, true),
  );
  if (shape[0] !== 64 || shape[3] !== 10 ||
      shape[1] < 4 || shape[1] > 256 ||
      shape[2] < 4 || shape[2] > 256) {
    return null;
  }
  let parameterCount = 0;
  for (let layer = 0; layer < shape.length - 1; layer += 1) {
    parameterCount += shape[layer] * shape[layer + 1] + shape[layer + 1];
  }
  const expectedBytes = 12 + MODEL_LAYER_COUNT * 4 + parameterCount * 8;
  return expectedBytes === model.byteLength ? shape : null;
}

function validStoredRecord(record) {
  if (!record || record.version !== MODEL_RECORD_VERSION ||
      typeof record.revision !== "string" || record.revision.length < 8 ||
      !record.config) {
    return false;
  }
  const { firstHidden, secondHidden, epochs, learningRate } = record.config;
  const shape = serializedModelShape(record.model);
  return shape !== null && shape[1] === firstHidden && shape[2] === secondHidden &&
    Number.isInteger(firstHidden) && firstHidden >= 4 && firstHidden <= 256 &&
    Number.isInteger(secondHidden) && secondHidden >= 4 && secondHidden <= 256 &&
    Number.isInteger(epochs) && epochs >= 10 && epochs <= 100 &&
    [0.03, 0.05, 0.08].includes(learningRate);
}

async function discardInvalidStoredModel(error, revision) {
  console.warn("Discarding stored model", error);
  removeVirtualModelFile();
  try {
    const removed = await deleteStoredModelRevision(revision);
    if (!removed) {
      ui.storageStatus.textContent =
        "Saved model changed in another tab · reload to inspect it";
      state.storedModelLoaded = false;
      return;
    }
  } catch (storageError) {
    console.warn("Could not delete invalid stored model", storageError);
  }
  state.storedModelLoaded = false;
  ui.storageStatus.textContent = "No saved model · a validated run will be kept locally";
}

function restoreDefaultModelAfterLoadFailure() {
  ui.architecture.value = "128,64";
  ui.epochs.value = "25";
  ui.learningRate.value = "0.05";
  updateConfigurationCopy();
  if (state.wasm._web_configure_model(128, 64, TRAINING_SEED) !== 0) {
    throw new Error("C runtime could not restore the default model");
  }
}

async function restoreStoredModel() {
  let record;
  try {
    record = await readStoredModel();
  } catch (error) {
    state.storageAvailable = false;
    ui.storageStatus.textContent = "Current tab only · browser storage unavailable";
    return false;
  }
  if (record === undefined) {
    ui.storageStatus.textContent = "No saved model · a validated run will be kept locally";
    return false;
  }
  if (!validStoredRecord(record)) {
    await discardInvalidStoredModel(
      new Error("Stored record has an invalid format"),
      record?.revision,
    );
    return false;
  }

  try {
    state.wasm.FS.writeFile(MODEL_PATH, new Uint8Array(record.model));
    const loaded = state.wasm.ccall(
      "web_load_model",
      "number",
      ["string"],
      [MODEL_PATH],
    );
    if (loaded !== 0) throw new Error("C rejected the stored model");
    const expectedShape = [
      64,
      record.config.firstHidden,
      record.config.secondHidden,
      10,
    ];
    if (modelShape().join(",") !== expectedShape.join(",")) {
      throw new Error("Stored model architecture does not match its metadata");
    }
    if (state.wasm._web_evaluate() !== 0) {
      throw new Error("Stored model could not be evaluated");
    }
    const testAccuracy = state.wasm._web_accuracy();
    const testLoss = state.wasm._web_loss();
    const drawingMetrics = await evaluateDrawingChecks();
    if (!qualityPassed(testAccuracy, drawingMetrics)) {
      throw new Error("Stored model no longer passes the quality gate");
    }

    const architecture = `${record.config.firstHidden},${record.config.secondHidden}`;
    if (!Array.from(ui.architecture.options).some((option) => option.value === architecture)) {
      throw new Error("Stored architecture is not offered by this interface");
    }
    ui.architecture.value = architecture;
    ui.epochs.value = String(record.config.epochs);
    ui.learningRate.value = String(record.config.learningRate);
    updateConfigurationCopy();
    state.currentConfig = { ...record.config };
    state.configured = true;
    state.modelTrained = true;
    state.storedModelLoaded = true;
    ui.testAccuracy.textContent = `${(testAccuracy * 100).toFixed(1)}%`;
    ui.testLoss.textContent = testLoss.toFixed(3);
    ui.drawingAccuracy.textContent = `${(drawingMetrics.accuracy * 100).toFixed(1)}%`;
    ui.trainingProgress.max = record.config.epochs;
    ui.trainingProgress.value = record.config.epochs;
    ui.epochLabel.textContent = `Saved after ${record.config.epochs} epochs`;
    setRecognitionEnabled(true);
    setBadge(ui.runtimeStatus, "Saved model loaded", "ready");
    ui.trainingStatus.textContent =
      "Loaded the last validated model stored by this browser profile.";
    const date = new Date(record.savedAt);
    ui.storageStatus.textContent = Number.isFinite(date.getTime())
      ? `Saved locally ${date.toLocaleString()}`
      : "Saved locally in this browser profile";
    renderNetworkAtRest("Saved model · weights");
    return true;
  } catch (error) {
    await discardInvalidStoredModel(error, record.revision);
    restoreDefaultModelAfterLoadFailure();
    return false;
  } finally {
    removeVirtualModelFile();
  }
}

async function initialize() {
  try {
    setupInterface();
    networkRenderer = new DenseNetworkRenderer(ui.networkCanvas);
    initializeProbabilityRows();
    bindEvents();
    updateConfigurationCopy();
    switchTab("train");
    setRecognitionEnabled(false);
    resetPrediction();
    window.__enenAppStarted = true;

    if (typeof createEnenModule !== "function") {
      throw new Error("The Emscripten loader is missing; run make web");
    }
    const [wasmBinary, trainingBytes, testBytes] = await Promise.all([
      fetchBytes(WASM_URL),
      fetchBytes(TRAIN_DATA_URL),
      fetchBytes(TEST_DATA_URL),
    ]);
    const wasm = await withTimeout(
      createEnenModule({
        locateFile: (path) => new URL(path, window.location.href).href,
        printErr: (message) => console.error(`[C/Wasm] ${message}`),
        wasmBinary,
      }),
      "The C / WebAssembly runtime did not start within 15 seconds",
    );
    state.wasm = wasm;
    wasm.FS.writeFile(TRAIN_DATA_PATH, trainingBytes);
    wasm.FS.writeFile(TEST_DATA_PATH, testBytes);

    const initialized = wasm.ccall(
      "web_initialize",
      "number",
      ["string", "string"],
      [TRAIN_DATA_PATH, TEST_DATA_PATH],
    );
    if (initialized !== 0) {
      throw new Error("C runtime rejected the bundled digit data");
    }
    await prepareDrawingData();

    const total = wasm._web_training_samples() + wasm._web_synthetic_samples();
    state.readyForTraining = true;
    setBadge(ui.runtimeStatus, "Checking local model");
    ui.trainingStatus.textContent = "Checking this browser for its last validated model…";
    const restored = await restoreStoredModel();
    if (!restored) {
      state.currentConfig = null;
      state.configured = false;
      state.modelTrained = false;
      setRecognitionEnabled(false);
      setBadge(ui.runtimeStatus, "C / Wasm ready", "ready");
      ui.trainingStatus.textContent =
        `Ready to train on ${total.toLocaleString()} normalized examples.`;
      ui.trainingProgress.value = 0;
      ui.epochLabel.textContent = `0 / ${ui.epochs.value} epochs`;
      renderNetworkAtRest("Fresh model · weights");
    }
    ui.trainButton.textContent = restored ? "Train a new model" : "Train model";
    setSetupControlsDisabled(false);
  } catch (error) {
    showStartupError(error);
  }
}

function configurationMatches(config) {
  return state.currentConfig &&
    state.currentConfig.firstHidden === config.firstHidden &&
    state.currentConfig.secondHidden === config.secondHidden;
}

function resetTrainingPresentation(config) {
  ui.trainingProgress.max = config.epochs;
  ui.trainingProgress.value = 0;
  ui.trainingProgress.setAttribute(
    "aria-label",
    `Training progress for ${config.epochs} epochs`,
  );
  ui.epochLabel.textContent = `0 / ${config.epochs} epochs`;
  ui.networkShape.textContent =
    `64 → ${config.firstHidden} → ${config.secondHidden} → 10`;
  ui.layerSummary.textContent =
    `64 → ${config.firstHidden} → ${config.secondHidden} → 10`;
  ui.networkSampleLabel.textContent = "—";
  ui.networkSampleCaption.textContent = "Waiting for the first C update";
  ui.visualizationStatus.textContent = "Starting the first training batch…";
  ui.completionActions.hidden = true;
  ui.goRecognizeButton.hidden = false;
  ui.pauseAnimationButton.hidden = false;
  ui.skipAnimationButton.hidden = false;
  ui.pauseAnimationButton.setAttribute("aria-pressed", "false");
  ui.pauseAnimationButton.textContent = "Pause animation";
  ui.pauseAnimationButton.disabled = false;
  ui.skipAnimationButton.disabled = false;
  ui.skipAnimationButton.textContent = "Finish without animation";
  state.animationPaused = false;
  state.skipAnimation = false;
  resetMetrics();
}

function configureSelectedModel({ announce = true } = {}) {
  if (!state.wasm || state.training || !state.readyForTraining) return false;
  const config = selectedConfiguration();
  const seed = state.nextSeed++;
  if (state.wasm._web_configure_model(
    config.firstHidden,
    config.secondHidden,
    seed,
  ) !== 0) {
    throw new Error("C runtime could not create the selected architecture");
  }
  const actualShape = Array.from(
    { length: state.wasm._web_num_layers() },
    (_, layer) => state.wasm._web_layer_size(layer),
  );
  const expectedShape = [64, config.firstHidden, config.secondHidden, 10];
  if (actualShape.join(",") !== expectedShape.join(",")) {
    throw new Error("C runtime created a different architecture than requested");
  }
  state.currentConfig = { ...config, seed };
  state.configured = true;
  state.modelTrained = false;
  setRecognitionEnabled(false);
  clearDrawing();
  resetTrainingPresentation(config);
  setBadge(ui.runtimeStatus, "Fresh C model", "ready");
  setBadge(ui.modelStatus, "Fresh weights");
  ui.storageStatus.textContent = state.storedModelLoaded
    ? "Fresh model in this tab · the last validated model remains saved"
    : "Fresh model · it will be saved only after validation";
  if (announce) {
    ui.trainingStatus.textContent =
      `Fresh ${actualShape.join(" → ")} model created with seed ${seed}.`;
  }
  return true;
}

function createFreshModel() {
  try {
    configureSelectedModel();
  } catch (error) {
    console.error(error);
    setBadge(ui.runtimeStatus, "Model failed", "error");
    ui.trainingStatus.textContent = error.message;
  }
}

async function trainModel(event) {
  event?.preventDefault();
  if (!state.wasm || state.training || !state.readyForTraining) return;
  const config = selectedConfiguration();
  try {
    if (!state.configured || state.modelTrained || !configurationMatches(config)) {
      configureSelectedModel({ announce: false });
    }
  } catch (error) {
    console.error(error);
    setBadge(ui.runtimeStatus, "Model failed", "error");
    ui.trainingStatus.textContent = error.message;
    return;
  }
  state.training = true;
  setSetupControlsDisabled(true);
  setRecognitionEnabled(false);
  resetPrediction();
  resetTrainingPresentation(config);
  showTrainingStage(true);
  setBadge(ui.runtimeStatus, "Training in C");
  setBadge(ui.modelStatus, "Training");
  ui.trainingAnnouncement.textContent =
    `Training started for ${config.epochs} epochs.`;
  await nextPaint();
  ui.trainingStageTitle.focus();

  try {
    state.currentConfig = { ...state.currentConfig, ...config };
    const totalSamples =
      state.wasm._web_training_samples() + state.wasm._web_synthetic_samples();
    renderTrainingSnapshot(true, false);
    ui.visualizationStatus.textContent =
      "Fresh weights responding to the first real training example";
    let nextVisualizationAt = performance.now() + VISUALIZATION_INTERVAL_MS;
    let lastAnnouncedEpoch = -1;

    while (state.wasm._web_epochs_trained() < config.epochs) {
      const fast = state.skipAnimation || state.reducedMotion;
      const batchSize = fast ? FAST_TRAINING_BATCH_SIZE : TRAINING_BATCH_SIZE;
      const processed = state.wasm._web_train_batch(batchSize, config.learningRate);
      if (processed <= 0) {
        const epoch = state.wasm._web_epochs_trained() + 1;
        throw new Error(`C training failed during epoch ${epoch}`);
      }

      const completed = state.wasm._web_epochs_trained();
      const position = state.wasm._web_epoch_position();
      const partial = position / totalSamples;
      ui.trainingProgress.value = Math.min(config.epochs, completed + partial);
      ui.epochLabel.textContent = `${completed} / ${config.epochs} epochs`;
      ui.visualizationStatus.textContent = completed < config.epochs
        ? `C is training epoch ${completed + 1} · ${position.toLocaleString()} / ${totalSamples.toLocaleString()} examples`
        : "Training updates complete · preparing final checks";

      const now = performance.now();
      if (!state.animationPaused && !fast && now >= nextVisualizationAt) {
        renderTrainingSnapshot(false, true);
        nextVisualizationAt = now + VISUALIZATION_INTERVAL_MS;
      }
      if (completed !== lastAnnouncedEpoch &&
          (completed === 1 || completed % 5 === 0 || completed === config.epochs)) {
        ui.trainingAnnouncement.textContent =
          `${completed} of ${config.epochs} training epochs complete.`;
        lastAnnouncedEpoch = completed;
      }

      if (!fast || position === 0) await nextPaint();
    }

    state.modelTrained = true;
    renderTrainingSnapshot(false, !state.skipAnimation && !state.reducedMotion);
    ui.trainingProgress.value = config.epochs;
    ui.epochLabel.textContent = `${config.epochs} / ${config.epochs} epochs`;
    ui.visualizationStatus.textContent = "Running the final held-out checks…";
    ui.trainingAnnouncement.textContent = "Training updates complete. Running final checks.";
    await nextPaint();
    if (state.wasm._web_evaluate() !== 0) {
      throw new Error("C evaluation failed");
    }
    const testAccuracy = state.wasm._web_accuracy();
    const testLoss = state.wasm._web_loss();
    const drawingMetrics = await evaluateDrawingChecks();

    ui.testAccuracy.textContent = `${(testAccuracy * 100).toFixed(1)}%`;
    ui.testLoss.textContent = testLoss.toFixed(3);
    ui.drawingAccuracy.textContent = `${(drawingMetrics.accuracy * 100).toFixed(1)}%`;

    if (!qualityPassed(testAccuracy, drawingMetrics)) {
      throw new Error(
        "Training completed, but this recipe did not meet the recognition quality checks. Adjust it and try again.",
      );
    }

    await persistValidatedModel({
      testAccuracy,
      testLoss,
      drawingAccuracy: drawingMetrics.accuracy,
      minimumClassAccuracy: drawingMetrics.minimumClassAccuracy,
      epochs: config.epochs,
    });
    setBadge(ui.runtimeStatus, "Model trained", "ready");
    setRecognitionEnabled(true);
    ui.visualizationStatus.textContent =
      "Training passed both checks. This exact C model is ready to recognize.";
    ui.trainingAnnouncement.textContent =
      "Training complete. The model passed its checks and recognition is ready.";
    if (state.hasInk) recognizeDrawing();
    else resetPrediction();
    ui.completionActions.hidden = false;
  } catch (error) {
    console.error(error);
    state.modelTrained = true;
    setRecognitionEnabled(false);
    setBadge(ui.runtimeStatus, "Training failed", "error");
    setBadge(ui.modelStatus, "Not ready", "error");
    setBadge(ui.recognitionModelStatus, "Not ready", "error");
    ui.visualizationStatus.textContent = error.message;
    ui.trainingAnnouncement.textContent = error.message;
    ui.goRecognizeButton.hidden = true;
    ui.completionActions.hidden = false;
  } finally {
    state.training = false;
    ui.pauseAnimationButton.hidden = true;
    ui.skipAnimationButton.hidden = true;
    setSetupControlsDisabled(false);
  }
}

async function evaluateDrawingChecks() {
  let correct = 0;
  const classCorrect = new Uint16Array(10);
  const classTotal = new Uint16Array(10);
  for (let index = 0; index < state.drawingChecks.length; index += 1) {
    const sample = state.drawingChecks[index];
    writeWasmInput(sample.pixels);
    const sampleCorrect = state.wasm._web_predict() === sample.label;
    correct += sampleCorrect;
    classCorrect[sample.label] += sampleCorrect;
    classTotal[sample.label] += 1;
    if (index > 0 && index % 40 === 0) {
      await nextPaint();
    }
  }
  const classAccuracies = Array.from(
    classCorrect,
    (count, digit) => count / classTotal[digit],
  );
  return {
    accuracy: correct / state.drawingChecks.length,
    minimumClassAccuracy: Math.min(...classAccuracies),
  };
}

function recognizeDrawing(announce = false) {
  if (!state.modelReady || !state.hasInk) {
    resetPrediction();
    return;
  }
  const pixels = canvasToPixels(drawingMemoryCanvas);
  if (!pixels) {
    setCanvasInkState(false);
    if (state.wasm && !state.training) {
      renderNetworkAtRest("Trained weights · no input");
    }
    resetPrediction();
    if (announce) ui.predictionAnnouncement.textContent = "Canvas empty.";
    return;
  }
  writeWasmInput(pixels);
  const prediction = state.wasm._web_predict();
  if (prediction < 0) {
    return;
  }
  const probabilities = Array.from(
    { length: 10 },
    (_, digit) => state.wasm._web_probability(digit),
  );
  const confidence = probabilities[prediction];
  ui.predictionDigit.textContent = String(prediction);
  ui.predictionConfidence.textContent = `${(confidence * 100).toFixed(1)}% confidence`;
  if (announce) {
    ui.predictionAnnouncement.textContent =
      `Predicted ${prediction} with ${(confidence * 100).toFixed(1)} percent confidence.`;
  }

  document.querySelectorAll(".probability-row").forEach((row, digit) => {
    const probability = probabilities[digit];
    row.classList.toggle("is-winner", digit === prediction);
    row.querySelector(".probability-track span").style.width = `${probability * 100}%`;
    row.querySelector(".probability-value").textContent = `${Math.round(probability * 100)}%`;
  });

  const now = performance.now();
  if (announce ||
      now - state.networkVisualizationAt >=
        RECOGNITION_VISUALIZATION_INTERVAL_MS) {
    try {
      renderRecognitionSnapshot(pixels);
    } catch (error) {
      console.error(error);
      ui.networkStatus.textContent =
        `Activation view unavailable: ${error.message}`;
    }
  }
}

function queueRecognition(announce = false) {
  state.announcePrediction ||= announce;
  if (state.recognitionFrame) return;
  state.recognitionFrame = requestAnimationFrame(() => {
    state.recognitionFrame = 0;
    const shouldAnnounce = state.announcePrediction;
    state.announcePrediction = false;
    recognizeDrawing(shouldAnnounce);
  });
}

function resetPrediction() {
  ui.predictionDigit.textContent = "—";
  ui.predictionConfidence.textContent = state.modelReady
    ? "Draw a digit to begin."
    : "Train the model first.";
  ui.predictionAnnouncement.textContent = "";
  document.querySelectorAll(".probability-row").forEach((row) => {
    row.classList.remove("is-winner");
    row.querySelector(".probability-track span").style.width = "0%";
    row.querySelector(".probability-value").textContent = "0%";
  });
}

function drawingPoint(event) {
  const bounds = ui.drawingCanvas.getBoundingClientRect();
  const x = (event.clientX - bounds.left) * (ui.drawingCanvas.width / bounds.width);
  const y = (event.clientY - bounds.top) * (ui.drawingCanvas.height / bounds.height);
  return {
    x: Math.max(0, Math.min(ui.drawingCanvas.width, x)),
    y: Math.max(0, Math.min(ui.drawingCanvas.height, y)),
  };
}

function strokeDrawingSegment(from, to) {
  applyDrawingTool(state.strokeTool);
  drawingContext.beginPath();
  drawingContext.moveTo(from.x, from.y);
  drawingContext.lineTo(to.x, to.y);
  drawingContext.stroke();
  drawingMemoryContext.beginPath();
  drawingMemoryContext.moveTo(
    from.x + DRAWING_MEMORY_PADDING,
    from.y + DRAWING_MEMORY_PADDING,
  );
  drawingMemoryContext.lineTo(
    to.x + DRAWING_MEMORY_PADDING,
    to.y + DRAWING_MEMORY_PADDING,
  );
  drawingMemoryContext.stroke();
}

function drawingMemoryHasInk() {
  const alpha = drawingMemoryContext.getImageData(
    0,
    0,
    drawingMemoryCanvas.width,
    drawingMemoryCanvas.height,
  ).data;
  for (let index = 3; index < alpha.length; index += 4) {
    if (alpha[index] > 10) return true;
  }
  return false;
}

function beginDrawing(event) {
  if (!state.modelReady || state.drawing ||
      (event.button !== undefined && event.button !== 0)) return;
  event.preventDefault();
  state.drawing = true;
  state.activePointerId = event.pointerId;
  state.strokeTool = state.drawingTool;
  if (state.strokeTool === "draw") setCanvasInkState(true);
  ui.drawingCanvas.setPointerCapture?.(event.pointerId);
  const point = drawingPoint(event);
  state.lastDrawingPoint = point;
  strokeDrawingSegment(point, { x: point.x + 0.01, y: point.y + 0.01 });
  queueRecognition();
}

function continueDrawing(event) {
  if (!state.drawing || event.pointerId !== state.activePointerId) return;
  event.preventDefault();
  const coalesced = event.getCoalescedEvents?.();
  const events = coalesced?.length ? coalesced : [event];
  for (const currentEvent of events) {
    const point = drawingPoint(currentEvent);
    strokeDrawingSegment(state.lastDrawingPoint, point);
    state.lastDrawingPoint = point;
  }
  queueRecognition();
}

function endDrawing(event) {
  if (!state.drawing || event.pointerId !== state.activePointerId) return;
  const finishedTool = state.strokeTool;
  state.drawing = false;
  state.activePointerId = null;
  state.lastDrawingPoint = null;
  if (event.pointerId !== undefined) {
    ui.drawingCanvas.releasePointerCapture?.(event.pointerId);
  }
  applyDrawingTool();
  if (finishedTool === "erase") {
    const hasInk = drawingMemoryHasInk();
    setCanvasInkState(hasInk);
    if (!hasInk) {
      cancelAnimationFrame(state.recognitionFrame);
      state.recognitionFrame = 0;
      state.announcePrediction = false;
      setDrawingTool("draw");
      renderNetworkAtRest("Trained weights · no input");
      resetPrediction();
      ui.predictionAnnouncement.textContent = "Canvas empty.";
      return;
    }
  }
  queueRecognition(true);
}

function clearDrawing({ announce = false } = {}) {
  if (state.activePointerId !== null) {
    try {
      ui.drawingCanvas.releasePointerCapture?.(state.activePointerId);
    } catch {
      // Pointer capture may already have ended.
    }
  }
  state.drawing = false;
  state.activePointerId = null;
  state.lastDrawingPoint = null;
  cancelAnimationFrame(state.recognitionFrame);
  state.recognitionFrame = 0;
  state.announcePrediction = false;
  drawingContext.clearRect(0, 0, ui.drawingCanvas.width, ui.drawingCanvas.height);
  drawingMemoryContext.clearRect(
    0,
    0,
    drawingMemoryCanvas.width,
    drawingMemoryCanvas.height,
  );
  setCanvasInkState(false);
  setDrawingTool("draw");
  if (state.wasm && !state.training) {
    renderNetworkAtRest(state.modelReady ? "Trained weights · no input" : "Fresh weights · no input");
  }
  resetPrediction();
  if (announce) ui.predictionAnnouncement.textContent = "Canvas cleared.";
}

function drawExample() {
  if (!state.modelReady) return;
  setDrawingTool("draw");
  const digit = state.exampleIndex % 10;
  const random = new SeededRandom(0x5f3759df + state.exampleIndex * 97);
  state.exampleIndex += 1;
  renderSyntheticDigit(ui.drawingCanvas, digit, random, true);
  drawingMemoryContext.clearRect(
    0,
    0,
    drawingMemoryCanvas.width,
    drawingMemoryCanvas.height,
  );
  drawingMemoryContext.drawImage(
    ui.drawingCanvas,
    DRAWING_MEMORY_PADDING,
    DRAWING_MEMORY_PADDING,
  );
  setCanvasInkState(true);
  recognizeDrawing(true);
}

const C_KEYWORDS = new Set([
  "break", "case", "const", "continue", "default", "do", "else", "enum",
  "extern", "for", "goto", "if", "register", "return", "sizeof", "static",
  "struct", "switch", "typedef", "union", "volatile", "while",
]);
const C_TYPES = new Set([
  "char", "double", "float", "int", "long", "short", "signed", "unsigned",
  "void", "size_t", "uint32_t", "Net", "Config", "ClassificationMetrics",
]);

function appendSourceToken(fragment, text, className = "") {
  if (!text) return;
  if (!className) {
    fragment.append(document.createTextNode(text));
    return;
  }
  const span = document.createElement("span");
  span.className = className;
  span.textContent = text;
  fragment.append(span);
}

function highlightCSource(source) {
  const fragment = document.createDocumentFragment();
  let index = 0;
  let lineStart = true;
  while (index < source.length) {
    if (lineStart) {
      const directive = source.slice(index).match(/^[ \t]*#[^\n]*(?:\n|$)/);
      if (directive) {
        appendSourceToken(fragment, directive[0], "tok-preprocessor");
        index += directive[0].length;
        lineStart = true;
        continue;
      }
    }
    if (source.startsWith("//", index)) {
      const end = source.indexOf("\n", index);
      const stop = end < 0 ? source.length : end + 1;
      appendSourceToken(fragment, source.slice(index, stop), "tok-comment");
      index = stop;
      lineStart = true;
      continue;
    }
    if (source.startsWith("/*", index)) {
      const end = source.indexOf("*/", index + 2);
      const stop = end < 0 ? source.length : end + 2;
      const token = source.slice(index, stop);
      appendSourceToken(fragment, token, "tok-comment");
      index = stop;
      lineStart = token.endsWith("\n");
      continue;
    }
    const character = source[index];
    if (character === '"' || character === "'") {
      const quote = character;
      let stop = index + 1;
      while (stop < source.length) {
        if (source[stop] === "\\") stop += 2;
        else if (source[stop++] === quote) break;
      }
      appendSourceToken(fragment, source.slice(index, stop), "tok-string");
      index = stop;
      lineStart = false;
      continue;
    }
    const identifier = source.slice(index).match(/^[A-Za-z_][A-Za-z0-9_]*/);
    if (identifier) {
      const word = identifier[0];
      const className = C_KEYWORDS.has(word)
        ? "tok-keyword"
        : C_TYPES.has(word) ? "tok-type" : "";
      appendSourceToken(fragment, word, className);
      index += word.length;
      lineStart = false;
      continue;
    }
    const number = source.slice(index).match(
      /^(?:0[xX][0-9A-Fa-f]+|(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)[uUlLfF]*/,
    );
    if (number) {
      appendSourceToken(fragment, number[0], "tok-number");
      index += number[0].length;
      lineStart = false;
      continue;
    }
    appendSourceToken(fragment, character);
    index += 1;
    lineStart = character === "\n";
  }
  ui.sourceCode.replaceChildren(fragment);
}

async function loadSource() {
  if (state.sourceLoaded || state.sourceLoading) return;
  state.sourceLoading = true;
  ui.sourceStatus.textContent = "Loading the canonical nn.c implementation…";
  try {
    const bytes = await fetchBytes(SOURCE_URL);
    const source = new TextDecoder().decode(bytes);
    highlightCSource(source);
    const lines = source.endsWith("\n")
      ? source.slice(0, -1).split("\n").length
      : source.split("\n").length;
    ui.sourceStatus.textContent =
      `nn.c · ${lines.toLocaleString()} lines · shown in full`;
    state.sourceLoaded = true;
  } catch (error) {
    console.error(error);
    ui.sourceCode.textContent = "Could not load nn.c.";
    ui.sourceStatus.textContent = error.message;
  } finally {
    state.sourceLoading = false;
  }
}

function openSourceDialog() {
  if (!ui.sourceDialog.open) ui.sourceDialog.showModal();
  void loadSource();
}

function toggleAnimationPause() {
  if (!state.training || state.skipAnimation) return;
  state.animationPaused = !state.animationPaused;
  ui.pauseAnimationButton.setAttribute(
    "aria-pressed",
    String(state.animationPaused),
  );
  ui.pauseAnimationButton.textContent = state.animationPaused
    ? "Resume animation"
    : "Pause animation";
  if (state.animationPaused) {
    networkRenderer?.freeze();
    ui.trainingAnnouncement.textContent =
      "Visualization paused. C training is still continuing.";
  } else {
    ui.trainingAnnouncement.textContent = "Visualization resumed.";
  }
}

function finishWithoutAnimation() {
  if (!state.training) return;
  state.skipAnimation = true;
  state.animationPaused = false;
  networkRenderer?.freeze();
  ui.pauseAnimationButton.disabled = true;
  ui.skipAnimationButton.disabled = true;
  ui.skipAnimationButton.textContent = "Finishing full training…";
  ui.trainingAnnouncement.textContent =
    "Animation stopped. All selected epochs and checks will still finish.";
}

function showConfiguration() {
  if (state.training) return;
  showTrainingStage(false);
  switchTab("train");
  ui.trainButton.textContent = "Train model";
  ui.trainingStatus.textContent = state.modelReady
    ? "The trained model is active. A new validated run will replace the local saved copy."
    : "Adjust the recipe, then train a fresh C model.";
  setSetupControlsDisabled(false);
  ui.architecture.focus();
}

function bindEvents() {
  for (const tab of [ui.trainTab, ui.recognizeTab]) {
    tab.addEventListener("click", () => switchTab(tab.dataset.tab));
    tab.addEventListener("keydown", handleTabKeydown);
  }
  ui.brand.addEventListener("click", (event) => {
    event.preventDefault();
    switchTab("train", true);
  });
  ui.sourceButton.addEventListener("click", openSourceDialog);
  ui.sourceCloseButton.addEventListener("click", () => ui.sourceDialog.close());
  ui.modelOptions.addEventListener("submit", (event) => void trainModel(event));
  ui.newModelButton.addEventListener("click", createFreshModel);
  ui.architecture.addEventListener("change", updateConfigurationCopy);
  ui.epochs.addEventListener("input", updateConfigurationCopy);
  ui.learningRate.addEventListener("change", updateConfigurationCopy);
  ui.pauseAnimationButton.addEventListener("click", toggleAnimationPause);
  ui.skipAnimationButton.addEventListener("click", finishWithoutAnimation);
  ui.configureAgainButton.addEventListener("click", showConfiguration);
  ui.goRecognizeButton.addEventListener("click", () => switchTab("recognize", true));
  ui.drawToolButton.addEventListener("click", () => setDrawingTool("draw"));
  ui.eraserToolButton.addEventListener("click", () => setDrawingTool("erase"));
  ui.clearButton.addEventListener("click", () => clearDrawing({ announce: true }));
  ui.exampleButton.addEventListener("click", drawExample);
  ui.drawingCanvas.addEventListener("pointerdown", beginDrawing);
  ui.drawingCanvas.addEventListener("pointermove", continueDrawing);
  ui.drawingCanvas.addEventListener("pointerup", endDrawing);
  ui.drawingCanvas.addEventListener("pointercancel", endDrawing);
}

function nextPaint() {
  return new Promise((resolve) => {
    let finished = false;
    let frame = 0;
    const finish = () => {
      if (finished) return;
      finished = true;
      clearTimeout(timeout);
      if (frame) cancelAnimationFrame(frame);
      resolve();
    };
    const timeout = setTimeout(finish, 32);
    if (typeof requestAnimationFrame === "function") {
      frame = requestAnimationFrame(finish);
    }
  });
}

void initialize();

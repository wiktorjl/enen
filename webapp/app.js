"use strict";

const MODEL_URL = "assets/digits.model";
const TRAIN_URL = "assets/optdigits.tra";
const TEST_URL = "assets/optdigits.tes";

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

const ui = {
  tabs: $$(".tab"),
  panels: $$(".tab-panel"),
  modelPill: $("#model-pill"),
  modelPillText: $("#model-pill-text"),
  architecture: $("#architecture"),
  epochs: $("#epochs"),
  epochsValue: $("#epochs-value"),
  learningRate: $("#learning-rate"),
  createModel: $("#create-model"),
  trainModel: $("#train-model"),
  trainButtonLabel: $("#train-button-label"),
  trainingMessage: $("#training-message"),
  epochCounter: $("#epoch-counter"),
  trainingProgress: $("#training-progress"),
  accuracyMetric: $("#accuracy-metric"),
  lossMetric: $("#loss-metric"),
  lossChart: $("#loss-chart"),
  chartCaption: $("#chart-caption"),
  drawingCanvas: $("#drawing-canvas"),
  drawHint: $("#draw-hint"),
  clearDrawing: $("#clear-drawing"),
  sampleDigit: $("#sample-digit"),
  predictionNumber: $("#prediction-number"),
  confidenceValue: $("#confidence-value"),
  confidenceFill: $("#confidence-fill"),
  probabilityList: $("#probability-list"),
  networkCanvas: $("#network-canvas"),
  networkEmpty: $("#network-empty"),
  layerSummary: $("#layer-summary"),
  networkEyebrow: $("#network-eyebrow"),
  networkNoteCopy: $("#network-note-copy"),
};

const state = {
  activeTab: "train",
  model: null,
  modelOrigin: "loading",
  training: false,
  history: [],
  trainData: null,
  testData: null,
  freehandData: null,
  dataPromise: null,
  trainingToken: 0,
  drawingActive: false,
  drawingHasInk: false,
  recognitionFrame: null,
  lastRecognitionModel: null,
  sampleCursor: 12,
  trainedEpochs: 0,
};

class SeededRandom {
  constructor(seed = 0x9e3779b9) {
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

class NeuralNetwork {
  constructor(layerSizes, randomize = true, seed = Date.now()) {
    this.layerSizes = [...layerSizes];
    this.weights = [];
    this.biases = [];
    this.activations = layerSizes.map((size) => new Float32Array(size));
    this.deltas = layerSizes.map((size) => new Float32Array(size));

    const random = new SeededRandom(seed);
    for (let layer = 0; layer < layerSizes.length - 1; layer += 1) {
      const fanIn = layerSizes[layer];
      const fanOut = layerSizes[layer + 1];
      const weights = new Float32Array(fanIn * fanOut);
      if (randomize) {
        const limit = Math.sqrt(6 / (fanIn + fanOut));
        for (let index = 0; index < weights.length; index += 1) {
          weights[index] = (random.next() * 2 - 1) * limit;
        }
      }
      this.weights.push(weights);
      this.biases.push(new Float32Array(fanOut));
    }
  }

  forward(input) {
    this.activations[0].set(input);
    const outputLayer = this.layerSizes.length - 1;

    for (let layer = 0; layer < outputLayer; layer += 1) {
      const previous = this.activations[layer];
      const current = this.activations[layer + 1];
      const weights = this.weights[layer];
      const biases = this.biases[layer];
      const nextSize = current.length;

      for (let output = 0; output < nextSize; output += 1) {
        let sum = biases[output];
        for (let inputIndex = 0; inputIndex < previous.length; inputIndex += 1) {
          sum += previous[inputIndex] * weights[inputIndex * nextSize + output];
        }
        current[output] = layer + 1 === outputLayer ? sum : stableSigmoid(sum);
      }
    }

    const output = this.activations[outputLayer];
    let largest = output[0];
    for (let index = 1; index < output.length; index += 1) {
      if (output[index] > largest) largest = output[index];
    }

    let total = 0;
    for (let index = 0; index < output.length; index += 1) {
      output[index] = Math.exp(output[index] - largest);
      total += output[index];
    }
    for (let index = 0; index < output.length; index += 1) {
      output[index] /= total;
    }
    return output;
  }

  trainOne(input, label, learningRate) {
    const probabilities = this.forward(input);
    const outputLayer = this.layerSizes.length - 1;
    const outputDeltas = this.deltas[outputLayer];

    for (let index = 0; index < outputDeltas.length; index += 1) {
      outputDeltas[index] = (index === label ? 1 : 0) - probabilities[index];
    }

    for (let layer = outputLayer - 1; layer >= 1; layer -= 1) {
      const current = this.activations[layer];
      const deltas = this.deltas[layer];
      const nextDeltas = this.deltas[layer + 1];
      const nextSize = this.layerSizes[layer + 1];
      const weights = this.weights[layer];

      for (let inputIndex = 0; inputIndex < current.length; inputIndex += 1) {
        let error = 0;
        const offset = inputIndex * nextSize;
        for (let output = 0; output < nextSize; output += 1) {
          error += nextDeltas[output] * weights[offset + output];
        }
        deltas[inputIndex] = error * current[inputIndex] * (1 - current[inputIndex]);
      }
    }

    for (let layer = 0; layer < outputLayer; layer += 1) {
      const previous = this.activations[layer];
      const nextDeltas = this.deltas[layer + 1];
      const nextSize = this.layerSizes[layer + 1];
      const weights = this.weights[layer];
      const biases = this.biases[layer];

      for (let inputIndex = 0; inputIndex < previous.length; inputIndex += 1) {
        const scaledInput = learningRate * previous[inputIndex];
        const offset = inputIndex * nextSize;
        for (let output = 0; output < nextSize; output += 1) {
          weights[offset + output] += scaledInput * nextDeltas[output];
        }
      }
      for (let output = 0; output < nextSize; output += 1) {
        biases[output] += learningRate * nextDeltas[output];
      }
    }

    return -Math.log(Math.max(probabilities[label], 1e-8));
  }

  static fromCModel(buffer) {
    const view = new DataView(buffer);
    let offset = 0;
    const readUint32 = () => {
      const value = view.getUint32(offset, true);
      offset += 4;
      return value;
    };
    const readInt32 = () => {
      const value = view.getInt32(offset, true);
      offset += 4;
      return value;
    };
    const readDouble = () => {
      const value = view.getFloat64(offset, true);
      offset += 8;
      return value;
    };

    if (readUint32() !== 0x4e454e45) throw new Error("Invalid model signature");
    if (readUint32() !== 1) throw new Error("Unsupported model version");
    const layerCount = readUint32();
    if (layerCount < 2 || layerCount > 16) throw new Error("Invalid model layers");

    const layerSizes = [];
    for (let layer = 0; layer < layerCount; layer += 1) {
      const size = readInt32();
      if (size < 1 || size > 4096) throw new Error("Invalid model layer size");
      layerSizes.push(size);
    }

    const network = new NeuralNetwork(layerSizes, false);
    for (const weights of network.weights) {
      for (let index = 0; index < weights.length; index += 1) {
        weights[index] = readDouble();
      }
    }
    for (const biases of network.biases) {
      for (let index = 0; index < biases.length; index += 1) {
        biases[index] = readDouble();
      }
    }
    if (offset !== buffer.byteLength) throw new Error("Unexpected data after model");
    return network;
  }
}

class NetworkRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.context = canvas.getContext("2d");
    this.network = null;
    this.resizeObserver = new ResizeObserver(() => this.draw(this.network));
    this.resizeObserver.observe(canvas);
  }

  draw(network) {
    this.network = network;
    const bounds = this.canvas.getBoundingClientRect();
    if (!bounds.width || !bounds.height) return;

    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const pixelWidth = Math.round(bounds.width * ratio);
    const pixelHeight = Math.round(bounds.height * ratio);
    if (this.canvas.width !== pixelWidth || this.canvas.height !== pixelHeight) {
      this.canvas.width = pixelWidth;
      this.canvas.height = pixelHeight;
    }

    const context = this.context;
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, bounds.width, bounds.height);
    if (!network) return;

    const width = bounds.width;
    const height = bounds.height;
    const compact = width < 520;
    const horizontalPadding = compact ? 34 : 48;
    const top = compact ? 54 : 62;
    const bottom = height - (compact ? 35 : 42);
    const columnGap = (width - horizontalPadding * 2) / (network.layerSizes.length - 1);
    const layouts = network.layerSizes.map((count, index) =>
      this.layoutLayer(count, horizontalPadding + index * columnGap, top, bottom, index, compact),
    );

    this.drawLabels(context, network, layouts, compact);
    this.drawConnections(context, network, layouts);
    this.drawNeurons(context, network, layouts, compact);
  }

  layoutLayer(count, centerX, top, bottom, layerIndex, compact) {
    if (layerIndex === 0) {
      const gap = compact ? 5.0 : 6.2;
      const size = compact ? 3.8 : 4.7;
      const positions = [];
      for (let index = 0; index < 64; index += 1) {
        const column = index % 8;
        const row = Math.floor(index / 8);
        positions.push({
          x: centerX + (column - 3.5) * gap,
          y: (top + bottom) / 2 + (row - 3.5) * gap,
          radius: size / 2,
          square: true,
        });
      }
      return positions;
    }

    if (layerIndex === state.model?.layerSizes.length - 1) {
      const gap = (bottom - top) / Math.max(9, count - 1);
      return Array.from({ length: count }, (_, index) => ({
        x: centerX,
        y: top + index * gap,
        radius: compact ? 5.3 : 6.6,
        output: true,
      }));
    }

    const columns = count > 96 ? 8 : count > 56 ? 6 : count > 28 ? 5 : 4;
    const rows = Math.ceil(count / columns);
    const xGap = compact ? 4.1 : 5.4;
    const yGap = Math.min((bottom - top) / Math.max(rows - 1, 1), compact ? 10.5 : 12.5);
    const positions = [];
    for (let index = 0; index < count; index += 1) {
      const column = index % columns;
      const row = Math.floor(index / columns);
      const itemsInLastRow = row === rows - 1 ? count - row * columns : columns;
      positions.push({
        x: centerX + (column - (itemsInLastRow - 1) / 2) * xGap,
        y: (top + bottom) / 2 + (row - (rows - 1) / 2) * yGap,
        radius: compact ? 1.65 : 2.15,
      });
    }
    return positions;
  }

  drawLabels(context, network, layouts, compact) {
    context.save();
    context.fillStyle = "rgba(232, 242, 236, .52)";
    context.textAlign = "center";
    context.font = `${compact ? 7 : 8}px DM Mono, monospace`;
    layouts.forEach((positions, index) => {
      const type = index === 0 ? "INPUT" : index === layouts.length - 1 ? "OUTPUT" : `HIDDEN ${index}`;
      const detail = index === 0 ? "8×8" : network.layerSizes[index];
      context.fillText(`${type} · ${detail}`, positions[0].x, compact ? 27 : 31);
    });
    context.restore();
  }

  drawConnections(context, network, layouts) {
    context.save();
    context.lineCap = "round";

    for (let layer = 0; layer < layouts.length - 1; layer += 1) {
      const sources = layouts[layer];
      const targets = layouts[layer + 1];
      const sampleCount = Math.min(42, Math.max(sources.length, targets.length));
      context.lineWidth = 0.55;
      context.strokeStyle = "rgba(203, 232, 216, .045)";
      for (let index = 0; index < sampleCount; index += 1) {
        const source = sources[(index * 17 + layer * 7) % sources.length];
        const target = targets[(index * 29 + layer * 11) % targets.length];
        drawCurve(context, source, target);
      }

      const sourceValues = network.activations[layer];
      const targetValues = network.activations[layer + 1];
      const hotSources = topIndices(sourceValues, layer === 0 ? 4 : 5);
      const hotTargets = topIndices(targetValues, 5);
      const weights = network.weights[layer];
      const targetCount = targets.length;

      for (const sourceIndex of hotSources) {
        for (const targetIndex of hotTargets) {
          const sourceActivation = normalizedActivation(sourceValues[sourceIndex], layer, false);
          const targetActivation = normalizedActivation(
            targetValues[targetIndex],
            layer + 1,
            layer + 1 === layouts.length - 1,
          );
          const weight = weights[sourceIndex * targetCount + targetIndex];
          const strength = Math.min(1, sourceActivation * targetActivation * (0.7 + Math.abs(weight)));
          if (strength < 0.08) continue;
          context.strokeStyle = weight >= 0
            ? `rgba(185, 245, 207, ${0.05 + strength * 0.42})`
            : `rgba(237, 115, 95, ${0.04 + strength * 0.3})`;
          context.lineWidth = 0.6 + strength * 1.15;
          drawCurve(context, sources[sourceIndex], targets[targetIndex]);
        }
      }
    }
    context.restore();
  }

  drawNeurons(context, network, layouts, compact) {
    const winner = indexOfMax(network.activations.at(-1));
    layouts.forEach((positions, layer) => {
      const outputLayer = layer === layouts.length - 1;
      positions.forEach((position, index) => {
        const raw = network.activations[layer][index];
        const activation = normalizedActivation(raw, layer, outputLayer);
        const isWinner = outputLayer && index === winner;

        context.save();
        if (activation > 0.72) {
          context.shadowBlur = 6 + activation * 8;
          context.shadowColor = isWinner ? "rgba(237,115,95,.65)" : "rgba(185,245,207,.65)";
        }
        context.fillStyle = isWinner
          ? mixColor([56, 65, 59], [237, 115, 95], Math.max(0.48, activation))
          : mixColor([39, 62, 54], [185, 245, 207], activation);
        context.strokeStyle = outputLayer ? "rgba(232, 242, 236, .4)" : "rgba(232, 242, 236, .12)";
        context.lineWidth = outputLayer ? 0.8 : 0.45;

        if (position.square) {
          const size = position.radius * 2;
          context.beginPath();
          context.roundRect(position.x - size / 2, position.y - size / 2, size, size, 1);
        } else {
          context.beginPath();
          context.arc(position.x, position.y, position.radius, 0, Math.PI * 2);
        }
        context.fill();
        context.stroke();
        context.restore();

        if (outputLayer) {
          context.save();
          context.fillStyle = isWinner ? "#ffad9e" : "rgba(232, 242, 236, .58)";
          context.font = `${isWinner ? "600" : "400"} ${compact ? 7 : 8}px DM Mono, monospace`;
          context.textBaseline = "middle";
          context.fillText(String(index), position.x + position.radius + (compact ? 5 : 7), position.y);
          context.restore();
        }
      });
    });
  }
}

function stableSigmoid(value) {
  if (value >= 0) return 1 / (1 + Math.exp(-value));
  const exponential = Math.exp(value);
  return exponential / (1 + exponential);
}

function indexOfMax(values) {
  let maximum = 0;
  for (let index = 1; index < values.length; index += 1) {
    if (values[index] > values[maximum]) maximum = index;
  }
  return maximum;
}

function topIndices(values, count) {
  return [...values.keys()]
    .sort((left, right) => values[right] - values[left])
    .slice(0, count);
}

function normalizedActivation(value, layer, isOutput) {
  if (layer === 0 || isOutput) return Math.max(0, Math.min(1, value));
  return Math.max(0, Math.min(1, (value - 0.16) / 0.84));
}

function mixColor(from, to, amount) {
  const value = Math.max(0, Math.min(1, amount));
  const channels = from.map((channel, index) => Math.round(channel + (to[index] - channel) * value));
  return `rgb(${channels.join(",")})`;
}

function drawCurve(context, source, target) {
  const middle = (source.x + target.x) / 2;
  context.beginPath();
  context.moveTo(source.x, source.y);
  context.bezierCurveTo(middle, source.y, middle, target.y, target.x, target.y);
  context.stroke();
}

const networkRenderer = new NetworkRenderer(ui.networkCanvas);

function updateModelSummary() {
  if (!state.model) return;
  const labels = state.model.layerSizes.map((size, index) => {
    const name = index === 0 ? "pixels" : index === state.model.layerSizes.length - 1 ? "digits" : "hidden";
    return `<span><b>${size}</b> ${name}</span>`;
  });
  ui.layerSummary.innerHTML = labels.join('<span class="summary-arrow">→</span>');
}

function setModelStatus(text, mode = "ready") {
  ui.modelPill.classList.toggle("is-ready", mode === "ready");
  ui.modelPill.classList.toggle("is-training", mode === "training");
  ui.modelPillText.textContent = text;
}

function showNetwork() {
  ui.networkEmpty.classList.add("is-hidden");
  networkRenderer.draw(state.model);
  updateModelSummary();
}

function createFreshModel(announce = true) {
  const hidden = ui.architecture.value.split(",").map(Number);
  state.model = new NeuralNetwork([64, ...hidden, 10], true, Date.now());
  state.modelOrigin = "fresh";
  state.lastRecognitionModel = null;
  state.history = [];
  state.trainedEpochs = 0;
  state.model.forward(new Float32Array(64));
  state.trainingToken += 1;
  drawLossChart();
  showNetwork();
  resetMetrics();
  setModelStatus("Fresh model", "ready");
  ui.trainingProgress.style.width = "0%";
  ui.epochCounter.textContent = "Not trained";
  if (announce) ui.trainingMessage.textContent = "Fresh weights initialized — ready to learn";
  if (state.drawingHasInk) recognizeDrawing();
}

async function loadPretrainedModel() {
  try {
    const response = await fetch(MODEL_URL);
    if (!response.ok) throw new Error(`Model request failed (${response.status})`);
    const loadedModel = NeuralNetwork.fromCModel(await response.arrayBuffer());
    if (state.modelOrigin !== "loading") return;
    state.model = loadedModel;
    state.modelOrigin = "pretrained";
    state.lastRecognitionModel = null;
    state.trainedEpochs = 25;
    state.model.forward(new Float32Array(64));
    setModelStatus("Pretrained · ready", "ready");
    ui.trainingMessage.textContent = "Pretrained model is ready";
    ui.epochCounter.textContent = "Ready";
    showNetwork();

    try {
      const [, testData] = await ensureData();
      if (state.model !== loadedModel || state.training) return;
      const example = testData[12];
      state.model.forward(example.pixels);
      networkRenderer.draw(state.model);
      const metrics = evaluate(state.model, testData);
      ui.accuracyMetric.textContent = `${(metrics.accuracy * 100).toFixed(1)}%`;
      ui.lossMetric.textContent = metrics.loss.toFixed(3);
    } catch (error) {
      console.warn("Model loaded, but evaluation data was unavailable:", error);
    }
  } catch (error) {
    console.warn("Could not load the bundled model:", error);
    createFreshModel(false);
    ui.trainingMessage.textContent = window.location.protocol === "file:"
      ? "Model created — run a local server to load the dataset"
      : "Fresh model created — train it to recognize digits";
    setModelStatus("Fresh model", "ready");
  }
}

async function ensureData() {
  if (state.trainData && state.testData) return [state.trainData, state.testData];
  if (!state.dataPromise) {
    state.dataPromise = Promise.all([
      fetch(TRAIN_URL).then(checkResponse).then((response) => response.text()),
      fetch(TEST_URL).then(checkResponse).then((response) => response.text()),
    ]).then(([trainText, testText]) => {
      state.trainData = parseDataset(trainText);
      state.testData = parseDataset(testText);
      return [state.trainData, state.testData];
    }).catch((error) => {
      state.dataPromise = null;
      throw error;
    });
  }
  return state.dataPromise;
}

function checkResponse(response) {
  if (!response.ok) throw new Error(`Dataset request failed (${response.status})`);
  return response;
}

function parseDataset(text) {
  const samples = [];
  for (const line of text.trim().split(/\r?\n/)) {
    if (!line) continue;
    const fields = line.split(",");
    if (fields.length !== 65) continue;
    const pixels = new Float32Array(64);
    for (let index = 0; index < 64; index += 1) {
      pixels[index] = Number(fields[index]) / 16;
    }
    samples.push({ pixels, label: Number(fields[64]) });
  }
  if (!samples.length) throw new Error("The bundled dataset is empty or malformed");
  return samples;
}

function createFreehandTrainingSamples(samplesPerDigit = 100) {
  const canvas = document.createElement("canvas");
  canvas.width = 80;
  canvas.height = 80;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  const random = new SeededRandom(0x3f4e454e);
  const samples = [];
  const fonts = ["Arial", "Georgia", "Verdana", "Times New Roman"];

  for (let label = 0; label < 10; label += 1) {
    for (let sampleIndex = 0; sampleIndex < samplesPerDigit; sampleIndex += 1) {
      context.clearRect(0, 0, 80, 80);
      context.save();
      context.translate(40 + (random.next() - 0.5) * 5, 40 + (random.next() - 0.5) * 4);
      context.rotate((random.next() - 0.5) * 0.16);
      context.scale(0.88 + random.next() * 0.18, 0.9 + random.next() * 0.15);

      if (sampleIndex % 2 === 0) {
        // The recognition preprocessor leaves roughly one 8×8 cell of
        // breathing room. Match that scale for the vector-drawn examples.
        context.scale(0.82, 0.82);
        context.strokeStyle = "#fff";
        context.lineWidth = 7.2 + random.next() * 3.3;
        context.lineCap = "round";
        context.lineJoin = "round";
        drawFreehandGlyph(context, label);
      } else {
        const font = fonts[Math.floor(random.next() * fonts.length)];
        const weight = random.next() < 0.4 ? "600" : "400";
        context.fillStyle = "#fff";
        context.font = `${weight} ${60 + Math.round(random.next() * 7)}px ${font}`;
        context.textAlign = "center";
        context.textBaseline = "alphabetic";
        const metrics = context.measureText(String(label));
        const verticalCenter = (metrics.actualBoundingBoxAscent - metrics.actualBoundingBoxDescent) / 2;
        context.fillText(String(label), 0, verticalCenter);
      }
      context.restore();

      const pixels = rasterToPixels(context.getImageData(0, 0, 80, 80).data, 80, 80);
      samples.push({ pixels, label, freehand: true });
    }
  }
  return samples;
}

function drawFreehandGlyph(context, digit) {
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
      break;
  }
  context.stroke();
}

function rasterToPixels(data, width, height) {
  const pixels = new Float32Array(64);
  const cellWidth = width / 8;
  const cellHeight = height / 8;
  for (let row = 0; row < 8; row += 1) {
    for (let column = 0; column < 8; column += 1) {
      let sum = 0;
      for (let y = 0; y < cellHeight; y += 1) {
        for (let x = 0; x < cellWidth; x += 1) {
          const pixelX = Math.floor(column * cellWidth + x);
          const pixelY = Math.floor(row * cellHeight + y);
          sum += data[(pixelY * width + pixelX) * 4 + 3] / 255;
        }
      }
      pixels[row * 8 + column] = Math.min(1, sum / (cellWidth * cellHeight * 0.72));
    }
  }
  return pixels;
}

async function trainModel() {
  if (state.training) return;
  const epochs = Number(ui.epochs.value);
  const learningRate = Number(ui.learningRate.value);
  const desiredHidden = ui.architecture.value.split(",").map(Number);
  const desiredShape = [64, ...desiredHidden, 10].join(",");

  if (!state.model || state.model.layerSizes.join(",") !== desiredShape || state.modelOrigin !== "fresh") {
    createFreshModel(false);
  }

  // Creating the requested architecture invalidates older work. Capture this
  // run's token only after that reset so this run does not cancel itself.
  state.training = true;
  const token = ++state.trainingToken;

  setTrainingControls(true);
  setModelStatus("Training in progress", "training");
  ui.trainingMessage.textContent = "Loading handwritten examples…";
  ui.epochCounter.textContent = `0 / ${epochs}`;
  ui.trainingProgress.style.width = "1%";

  try {
    const [trainData, testData] = await ensureData();
    if (token !== state.trainingToken) return;

    if (!state.freehandData) state.freehandData = createFreehandTrainingSamples();
    const trainingData = trainData.concat(state.freehandData);
    const order = new Uint32Array(trainingData.length);
    const augmentedInput = new Float32Array(64);
    for (let index = 0; index < order.length; index += 1) order[index] = index;
    const random = new SeededRandom(0x51f15e + epochs + desiredHidden[0]);
    state.history = [];
    drawLossChart();

    for (let epoch = 0; epoch < epochs; epoch += 1) {
      shuffle(order, random);
      let runningLoss = 0;
      const visualizationStride = epochs > 80 ? 600 : epochs > 30 ? 300 : 150;

      for (let position = 0; position < order.length; position += 1) {
        const sample = trainingData[order[position]];
        let trainingInput = sample.pixels;
        if (epoch > 0 && random.next() < 0.42) {
          const offsetX = (random.next() - 0.5) * 0.9;
          const offsetY = (random.next() - 0.5) * 0.7;
          translatePixelsInto(sample.pixels, augmentedInput, offsetX, offsetY);
          trainingInput = augmentedInput;
        }
        runningLoss += state.model.trainOne(trainingInput, sample.label, learningRate);

        if (position % visualizationStride === 0) {
          const overallProgress = (epoch + position / order.length) / epochs;
          ui.trainingProgress.style.width = `${Math.max(1, overallProgress * 100)}%`;
          ui.trainingMessage.textContent = `Learning from a handwritten ${sample.label}`;
          ui.epochCounter.textContent = `${epoch + 1} / ${epochs}`;
          networkRenderer.draw(state.model);
          await nextFrame();
          if (token !== state.trainingToken) return;
        }
      }

      ui.trainingMessage.textContent = "Checking the held-out test set…";
      await nextFrame();
      const metrics = evaluate(state.model, testData);
      const averageTrainLoss = runningLoss / order.length;
      state.history.push(averageTrainLoss);
      ui.accuracyMetric.textContent = `${(metrics.accuracy * 100).toFixed(1)}%`;
      ui.lossMetric.textContent = metrics.loss.toFixed(3);
      ui.chartCaption.textContent = `Epoch ${epoch + 1} · loss ${averageTrainLoss.toFixed(3)}`;
      ui.trainingProgress.style.width = `${((epoch + 1) / epochs) * 100}%`;
      drawLossChart();
      await nextFrame();
    }

    state.modelOrigin = "trained";
    state.trainedEpochs = epochs;
    state.trainingProgress = 1;
    ui.trainingProgress.style.width = "100%";
    ui.trainingMessage.textContent = "Training complete — try your handwriting";
    ui.epochCounter.textContent = `${epochs} / ${epochs}`;
    setModelStatus(`Trained ${epochs} ep · ready`, "ready");
    if (state.drawingHasInk) recognizeDrawing();
  } catch (error) {
    console.error(error);
    ui.trainingMessage.textContent = window.location.protocol === "file:"
      ? "Dataset blocked by the browser — serve this folder locally"
      : "Could not load the training dataset";
    ui.epochCounter.textContent = "See README";
    setModelStatus("Model ready", "ready");
  } finally {
    if (token === state.trainingToken) {
      state.training = false;
      setTrainingControls(false);
      networkRenderer.draw(state.model);
    }
  }
}

function shuffle(values, random) {
  for (let index = values.length - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(random.next() * (index + 1));
    const temporary = values[index];
    values[index] = values[swapIndex];
    values[swapIndex] = temporary;
  }
}

function evaluate(network, samples) {
  let correct = 0;
  let loss = 0;
  for (const sample of samples) {
    const probabilities = network.forward(sample.pixels);
    if (indexOfMax(probabilities) === sample.label) correct += 1;
    loss -= Math.log(Math.max(probabilities[sample.label], 1e-8));
  }
  return { accuracy: correct / samples.length, loss: loss / samples.length };
}

function setTrainingControls(disabled) {
  ui.architecture.disabled = disabled;
  ui.epochs.disabled = disabled;
  ui.learningRate.disabled = disabled;
  ui.createModel.disabled = disabled;
  ui.trainModel.disabled = disabled;
  ui.trainButtonLabel.textContent = disabled ? "Training…" : "Start training";
}

function resetMetrics() {
  ui.accuracyMetric.textContent = "—";
  ui.lossMetric.textContent = "—";
  ui.chartCaption.textContent = "Waiting for training";
}

function drawLossChart() {
  const canvas = ui.lossChart;
  const bounds = canvas.getBoundingClientRect();
  if (!bounds.width || !bounds.height) return;
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(bounds.width * ratio);
  canvas.height = Math.round(bounds.height * ratio);
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, bounds.width, bounds.height);

  context.strokeStyle = "rgba(105, 113, 109, .14)";
  context.lineWidth = 1;
  context.beginPath();
  context.moveTo(0, bounds.height - 5);
  context.lineTo(bounds.width, bounds.height - 5);
  context.stroke();

  if (state.history.length < 2) return;
  const minimum = Math.min(...state.history);
  const maximum = Math.max(...state.history);
  const range = Math.max(0.1, maximum - minimum);
  context.strokeStyle = "#ed735f";
  context.lineWidth = 1.7;
  context.lineJoin = "round";
  context.beginPath();
  state.history.forEach((loss, index) => {
    const x = 2 + (index / (state.history.length - 1)) * (bounds.width - 4);
    const y = 5 + ((maximum - loss) / range) * (bounds.height - 12);
    if (index === 0) context.moveTo(x, y);
    else context.lineTo(x, y);
  });
  context.stroke();
}

function switchTab(name, moveFocus = false) {
  state.activeTab = name;
  ui.tabs.forEach((tab) => {
    const active = tab.dataset.tab === name;
    tab.classList.toggle("is-active", active);
    tab.setAttribute("aria-selected", String(active));
    tab.tabIndex = active ? 0 : -1;
    if (active && moveFocus) tab.focus();
  });
  ui.panels.forEach((panel) => {
    panel.hidden = panel.id !== `${name}-panel`;
  });

  const recognizing = name === "recognize";
  ui.networkEyebrow.textContent = recognizing ? "Live recognition signal" : "Live training signal";
  ui.networkNoteCopy.textContent = recognizing
    ? "Every stroke is reduced to 64 pixels. Brighter neurons are responding most strongly right now."
    : "Brighter neurons are responding more strongly to the current training example.";
  requestAnimationFrame(() => {
    if (recognizing && state.drawingHasInk) recognizeDrawing();
    else networkRenderer.draw(state.model);
    if (!recognizing) drawLossChart();
  });
}

function initializeProbabilities() {
  ui.probabilityList.innerHTML = Array.from({ length: 10 }, (_, digit) => `
    <div class="probability-row" data-digit="${digit}">
      <span>${digit}</span>
      <span class="mini-track"><span style="width:0%"></span></span>
      <span class="probability-value">0%</span>
    </div>
  `).join("");
}

const drawingContext = ui.drawingCanvas.getContext("2d", { willReadFrequently: true });
drawingContext.lineCap = "round";
drawingContext.lineJoin = "round";
drawingContext.strokeStyle = "#f7f4e9";
drawingContext.lineWidth = 23;

function drawingPoint(event) {
  const bounds = ui.drawingCanvas.getBoundingClientRect();
  return {
    x: (event.clientX - bounds.left) * (ui.drawingCanvas.width / bounds.width),
    y: (event.clientY - bounds.top) * (ui.drawingCanvas.height / bounds.height),
  };
}

function beginDrawing(event) {
  if (event.button !== undefined && event.button !== 0) return;
  state.drawingActive = true;
  state.drawingHasInk = true;
  ui.drawHint.classList.add("is-hidden");
  ui.drawingCanvas.setPointerCapture?.(event.pointerId);
  const point = drawingPoint(event);
  drawingContext.beginPath();
  drawingContext.moveTo(point.x, point.y);
  drawingContext.lineTo(point.x + 0.01, point.y + 0.01);
  drawingContext.stroke();
  queueRecognition();
}

function continueDrawing(event) {
  if (!state.drawingActive) return;
  const point = drawingPoint(event);
  drawingContext.lineTo(point.x, point.y);
  drawingContext.stroke();
  queueRecognition();
}

function endDrawing(event) {
  if (!state.drawingActive) return;
  state.drawingActive = false;
  drawingContext.closePath();
  if (event?.pointerId !== undefined) ui.drawingCanvas.releasePointerCapture?.(event.pointerId);
  queueRecognition();
}

function queueRecognition() {
  if (state.recognitionFrame) return;
  state.recognitionFrame = requestAnimationFrame(() => {
    state.recognitionFrame = null;
    recognizeDrawing();
  });
}

function preprocessDrawing() {
  const width = ui.drawingCanvas.width;
  const height = ui.drawingCanvas.height;
  const image = drawingContext.getImageData(0, 0, width, height);
  let left = width;
  let right = -1;
  let top = height;
  let bottom = -1;

  for (let y = 0; y < height; y += 2) {
    for (let x = 0; x < width; x += 2) {
      const offset = (y * width + x) * 4;
      if (image.data[offset + 3] > 20) {
        left = Math.min(left, x);
        right = Math.max(right, x);
        top = Math.min(top, y);
        bottom = Math.max(bottom, y);
      }
    }
  }

  if (right < left || bottom < top) return new Float32Array(64);

  const sourceWidth = right - left + 1;
  const sourceHeight = bottom - top + 1;
  const normalizedCanvas = document.createElement("canvas");
  normalizedCanvas.width = 80;
  normalizedCanvas.height = 80;
  const context = normalizedCanvas.getContext("2d", { willReadFrequently: true });
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = "high";

  const scale = Math.min(60 / sourceWidth, 60 / sourceHeight);
  const targetWidth = sourceWidth * scale;
  const targetHeight = sourceHeight * scale;
  const targetX = (80 - targetWidth) / 2;
  const targetY = (80 - targetHeight) / 2;
  context.drawImage(
    ui.drawingCanvas,
    left,
    top,
    sourceWidth,
    sourceHeight,
    targetX,
    targetY,
    targetWidth,
    targetHeight,
  );

  const normalized = context.getImageData(0, 0, 80, 80).data;
  const pixels = new Float32Array(64);
  for (let row = 0; row < 8; row += 1) {
    for (let column = 0; column < 8; column += 1) {
      let sum = 0;
      for (let y = 0; y < 10; y += 1) {
        for (let x = 0; x < 10; x += 1) {
          const offset = (((row * 10 + y) * 80) + column * 10 + x) * 4;
          sum += normalized[offset + 3] / 255;
        }
      }
      pixels[row * 8 + column] = Math.min(1, sum / 70);
    }
  }
  return pixels;
}

function translatePixelsInto(input, output, offsetX, offsetY) {
  output.fill(0);
  for (let row = 0; row < 8; row += 1) {
    for (let column = 0; column < 8; column += 1) {
      const sourceX = column - offsetX;
      const sourceY = row - offsetY;
      const left = Math.floor(sourceX);
      const top = Math.floor(sourceY);
      const fractionX = sourceX - left;
      const fractionY = sourceY - top;
      let value = 0;

      for (let yStep = 0; yStep <= 1; yStep += 1) {
        const sourceRow = top + yStep;
        if (sourceRow < 0 || sourceRow >= 8) continue;
        const weightY = yStep ? fractionY : 1 - fractionY;
        for (let xStep = 0; xStep <= 1; xStep += 1) {
          const sourceColumn = left + xStep;
          if (sourceColumn < 0 || sourceColumn >= 8) continue;
          const weightX = xStep ? fractionX : 1 - fractionX;
          value += input[sourceRow * 8 + sourceColumn] * weightX * weightY;
        }
      }
      output[row * 8 + column] = value;
    }
  }
  return output;
}

function thickenPixelsInto(input, output) {
  for (let row = 0; row < 8; row += 1) {
    for (let column = 0; column < 8; column += 1) {
      const index = row * 8 + column;
      let neighbor = 0;
      if (column > 0) neighbor = Math.max(neighbor, input[index - 1]);
      if (column < 7) neighbor = Math.max(neighbor, input[index + 1]);
      if (row > 0) neighbor = Math.max(neighbor, input[index - 8]);
      if (row < 7) neighbor = Math.max(neighbor, input[index + 8]);
      output[index] = Math.min(1, Math.max(input[index], input[index] * 0.88 + neighbor * 0.12));
    }
  }
  return output;
}

function robustForward(network, pixels) {
  const variants = [pixels];
  const shifts = [
    [-0.42, 0],
    [0.42, 0],
    [0, -0.36],
    [0, 0.36],
  ];
  for (const [offsetX, offsetY] of shifts) {
    variants.push(translatePixelsInto(pixels, new Float32Array(64), offsetX, offsetY));
  }
  variants.push(thickenPixelsInto(pixels, new Float32Array(64)));

  const outputs = variants.map((variant) => Float32Array.from(network.forward(variant)));
  const average = new Float32Array(10);
  for (const output of outputs) {
    for (let digit = 0; digit < average.length; digit += 1) average[digit] += output[digit] / outputs.length;
  }

  const winner = indexOfMax(average);
  let representative = 0;
  for (let index = 1; index < outputs.length; index += 1) {
    if (outputs[index][winner] > outputs[representative][winner]) representative = index;
  }

  // Keep hidden activations from the alignment that best supports the final
  // class, while output neurons show the alignment-averaged probabilities.
  network.forward(variants[representative]);
  network.activations.at(-1).set(average);
  return network.activations.at(-1);
}

function recognizeDrawing(input = null) {
  if (!state.model) return;
  const recognitionModel = state.model;
  state.lastRecognitionModel = recognitionModel;
  const pixels = input || preprocessDrawing();
  if (!input && !state.drawingHasInk) {
    resetPrediction();
    return;
  }
  // This is the exact network instance trained in trainModel(). Dataset
  // examples use their canonical alignment; freehand input is evaluated over
  // a few small alignments to match the UCI capture process more reliably.
  const probabilities = input
    ? recognitionModel.forward(pixels)
    : robustForward(recognitionModel, pixels);
  const winner = indexOfMax(probabilities);
  const confidence = probabilities[winner];
  ui.predictionNumber.textContent = String(winner);
  ui.confidenceValue.textContent = `${(confidence * 100).toFixed(1)}%`;
  ui.confidenceFill.style.width = `${confidence * 100}%`;
  $$(".probability-row").forEach((row, digit) => {
    const value = probabilities[digit];
    row.classList.toggle("is-winner", digit === winner);
    row.querySelector(".mini-track span").style.width = `${value * 100}%`;
    row.querySelector(".probability-value").textContent = `${Math.round(value * 100)}%`;
  });
  networkRenderer.draw(recognitionModel);
}

function clearDrawing() {
  drawingContext.clearRect(0, 0, ui.drawingCanvas.width, ui.drawingCanvas.height);
  state.drawingHasInk = false;
  ui.drawHint.classList.remove("is-hidden");
  resetPrediction();
  if (state.model) {
    state.model.forward(new Float32Array(64));
    networkRenderer.draw(state.model);
  }
}

function resetPrediction() {
  ui.predictionNumber.textContent = "—";
  ui.confidenceValue.textContent = "Draw to begin";
  ui.confidenceFill.style.width = "0%";
  $$(".probability-row").forEach((row) => {
    row.classList.remove("is-winner");
    row.querySelector(".mini-track span").style.width = "0%";
    row.querySelector(".probability-value").textContent = "0%";
  });
}

async function drawSampleDigit() {
  try {
    const [, testData] = await ensureData();
    const sample = testData[state.sampleCursor % testData.length];
    state.sampleCursor += 137;
    drawingContext.clearRect(0, 0, ui.drawingCanvas.width, ui.drawingCanvas.height);
    const cell = ui.drawingCanvas.width / 8;
    for (let index = 0; index < sample.pixels.length; index += 1) {
      const value = sample.pixels[index];
      if (!value) continue;
      drawingContext.fillStyle = `rgba(247, 244, 233, ${value})`;
      drawingContext.fillRect((index % 8) * cell, Math.floor(index / 8) * cell, cell + 0.5, cell + 0.5);
    }
    state.drawingHasInk = true;
    ui.drawHint.classList.add("is-hidden");
    recognizeDrawing(sample.pixels);
  } catch (error) {
    console.error(error);
    ui.confidenceValue.textContent = "Dataset unavailable";
  }
}

function nextFrame() {
  return new Promise((resolve) => requestAnimationFrame(resolve));
}

function bindEvents() {
  ui.tabs.forEach((tab, index) => {
    tab.addEventListener("click", () => switchTab(tab.dataset.tab));
    tab.addEventListener("keydown", (event) => {
      if (!["ArrowLeft", "ArrowRight"].includes(event.key)) return;
      event.preventDefault();
      const direction = event.key === "ArrowRight" ? 1 : -1;
      const nextIndex = (index + direction + ui.tabs.length) % ui.tabs.length;
      switchTab(ui.tabs[nextIndex].dataset.tab, true);
    });
  });

  ui.epochs.addEventListener("input", () => {
    ui.epochsValue.textContent = ui.epochs.value;
  });
  ui.createModel.addEventListener("click", () => createFreshModel());
  ui.trainModel.addEventListener("click", trainModel);
  ui.clearDrawing.addEventListener("click", clearDrawing);
  ui.sampleDigit.addEventListener("click", drawSampleDigit);
  ui.drawingCanvas.addEventListener("pointerdown", beginDrawing);
  ui.drawingCanvas.addEventListener("pointermove", continueDrawing);
  ui.drawingCanvas.addEventListener("pointerup", endDrawing);
  ui.drawingCanvas.addEventListener("pointercancel", endDrawing);
  ui.drawingCanvas.addEventListener("pointerleave", (event) => {
    if (state.drawingActive && !ui.drawingCanvas.hasPointerCapture?.(event.pointerId)) endDrawing(event);
  });
  window.addEventListener("resize", () => drawLossChart());
}

function initialize() {
  initializeProbabilities();
  bindEvents();
  drawLossChart();
  loadPretrainedModel();
}

initialize();

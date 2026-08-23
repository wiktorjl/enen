import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";

const loaderSource = fs.readFileSync("webapp/enen.js", "utf8");
const createEnenModule = vm.runInThisContext(
  `${loaderSource}\ncreateEnenModule;`,
  { filename: "webapp/enen.js" },
);

const wasm = await createEnenModule({
  wasmBinary: fs.readFileSync("webapp/enen.wasm"),
  printErr: (message) => console.error(`[C/Wasm] ${message}`),
});

function modelShape() {
  return Array.from(
    { length: wasm._web_num_layers() },
    (_, layer) => wasm._web_layer_size(layer),
  );
}

function inspectOutputs(sampleIndex) {
  assert.ok(wasm._web_copy_test_sample(sampleIndex) >= 0);
  assert.ok(wasm._web_inspect_input() >= 0);
  const outputLayer = wasm._web_num_layers() - 1;
  return Array.from(
    { length: wasm._web_layer_size(outputLayer) },
    (_, digit) => wasm._web_activation(outputLayer, digit),
  );
}

function saveModel(path) {
  return wasm.ccall("web_save_model", "number", ["string"], [path]);
}

function loadModel(path) {
  return wasm.ccall("web_load_model", "number", ["string"], [path]);
}

function modelHeaderBytes(layerSizes) {
  const bytes = new Uint8Array(12 + layerSizes.length * 4);
  const view = new DataView(bytes.buffer);
  let offset = 0;
  for (const value of [0x4e454e45, 1, layerSizes.length]) {
    view.setUint32(offset, value, true);
    offset += 4;
  }
  for (const size of layerSizes) {
    view.setInt32(offset, size, true);
    offset += 4;
  }
  return bytes;
}

function zeroModelBytes(layerSizes) {
  const weightCount = layerSizes.slice(0, -1).reduce(
    (total, size, layer) => total + size * layerSizes[layer + 1],
    0,
  );
  const biasCount = layerSizes.slice(1).reduce(
    (total, size) => total + size,
    0,
  );
  const bytes = new Uint8Array(
    12 + layerSizes.length * 4 + (weightCount + biasCount) * 8,
  );
  bytes.set(modelHeaderBytes(layerSizes));
  return bytes;
}

const persistencePaths = [
  "/web-api-roundtrip.model",
  "/web-api-malformed.model",
  "/web-api-wrong-shape.model",
  "/web-api-oversized-header.model",
  "/web-api-truncated.model",
  "/web-api-trailing.model",
];

try {
  wasm.FS.writeFile(
    "/optdigits.tra",
    fs.readFileSync("webapp/assets/optdigits.tra"),
  );
  wasm.FS.writeFile(
    "/optdigits.tes",
    fs.readFileSync("webapp/assets/optdigits.tes"),
  );
  assert.equal(
    wasm.ccall(
      "web_initialize",
      "number",
      ["string", "string"],
      ["/optdigits.tra", "/optdigits.tes"],
    ),
    0,
    "Wasm initializes through the exported C facade",
  );
  assert.equal(wasm._web_training_samples(), 3823);
  assert.equal(wasm._web_test_samples(), 1797);
  const inputOffset = wasm._web_input_buffer() / Float64Array.BYTES_PER_ELEMENT;
  wasm.HEAPF64.fill(0, inputOffset, inputOffset + 64);
  wasm.HEAPF64[inputOffset + 27] = 0.75;
  assert.equal(wasm._web_add_synthetic_sample(4), 0);
  assert.equal(wasm._web_synthetic_samples(), 1);
  assert.equal(wasm._web_clear_synthetic_samples(), 0);
  assert.equal(wasm._web_synthetic_samples(), 0);
  for (let label = 0; label < 10; label += 1) {
    const currentOffset = wasm._web_input_buffer() / Float64Array.BYTES_PER_ELEMENT;
    wasm.HEAPF64.fill(0, currentOffset, currentOffset + 64);
    wasm.HEAPF64[currentOffset + 8 + label * 5] = 0.8;
    assert.equal(wasm._web_add_synthetic_sample(label), 0);
  }
  assert.equal(wasm._web_synthetic_samples(), 10);

  assert.equal(
    wasm._web_configure_model(24, 16, 11),
    0,
    "Wasm creates a user-selected C architecture",
  );
  assert.deepEqual(modelShape(), [64, 24, 16, 10]);
  assert.equal(wasm._web_activation_count(), 64 + 24 + 16 + 10);
  assert.equal(wasm._web_configure_model(3, 16, 5), -1);
  assert.deepEqual(
    modelShape(),
    [64, 24, 16, 10],
    "an undersized hidden layer leaves the active model intact",
  );
  assert.equal(wasm._web_configure_model(257, 16, 5), -1);
  assert.deepEqual(
    modelShape(),
    [64, 24, 16, 10],
    "an oversized hidden layer leaves the active model intact",
  );

  assert.equal(wasm._web_reset_model(11), 0);
  assert.deepEqual(modelShape(), [64, 24, 16, 10]);
  assert.equal(wasm._web_epochs_trained(), 0);
  assert.equal(wasm._web_epoch_position(), 0);
  assert.equal(wasm._web_activation_version(), 0);
  assert.equal(wasm._web_last_training_label(), -1);

  assert.equal(
    wasm._web_copy_training_sample(wasm._web_training_samples()),
    0,
    "Wasm copies a retained synthetic training sample",
  );
  assert.ok(wasm._web_inspect_input() >= 0);
  let activationCount = 0;
  let inspectedOutputSum = 0;
  for (let layer = 0; layer < wasm._web_num_layers(); layer += 1) {
    const layerSize = wasm._web_layer_size(layer);
    activationCount += layerSize;
    for (let node = 0; node < layerSize; node += 1) {
      const activation = wasm._web_activation(layer, node);
      assert.ok(
        Number.isFinite(activation) && activation >= 0 && activation <= 1,
        `layer ${layer} activation ${node} was ${activation}`,
      );
      if (layer === wasm._web_num_layers() - 1) inspectedOutputSum += activation;
    }
    if (layer < wasm._web_num_layers() - 1) {
      const weightsPointer = wasm._web_layer_weights(layer);
      const weightOffset = weightsPointer / Float64Array.BYTES_PER_ELEMENT;
      const weightCount = layerSize * wasm._web_layer_size(layer + 1);
      assert.ok(weightsPointer > 0, `layer ${layer} exposes its C weights`);
      assert.ok(Number.isFinite(wasm.HEAPF64[weightOffset]));
      assert.ok(Number.isFinite(wasm.HEAPF64[weightOffset + weightCount - 1]));
    }
  }
  assert.equal(activationCount, wasm._web_activation_count());
  assert.ok(Math.abs(inspectedOutputSum - 1) < 1e-9);
  assert.equal(wasm._web_layer_weights(wasm._web_num_layers() - 1), 0);

  const initialVersion = wasm._web_activation_version();
  assert.equal(wasm._web_train_batch(7, 0.05), 7);
  assert.equal(wasm._web_epoch_position(), 7);
  assert.equal(
    saveModel(persistencePaths[0]),
    -1,
    "Wasm refuses to save during a partial shuffled epoch",
  );
  assert.equal(wasm._web_activation_version(), initialVersion + 1);
  assert.ok(
    wasm._web_last_training_label() >= 0 &&
      wasm._web_last_training_label() < 10,
  );

  const snapshotPointer = wasm._web_activation_snapshot();
  const snapshotOffset = snapshotPointer / Float64Array.BYTES_PER_ELEMENT;
  const snapshot = Float64Array.from(
    wasm.HEAPF64.subarray(snapshotOffset, snapshotOffset + activationCount),
  );
  assert.ok(snapshotPointer > 0);
  for (const activation of snapshot) {
    assert.ok(Number.isFinite(activation) && activation >= 0 && activation <= 1);
  }
  let flattenedOffset = 0;
  for (let layer = 0; layer < wasm._web_num_layers(); layer += 1) {
    for (let node = 0; node < wasm._web_layer_size(layer); node += 1) {
      assert.equal(
        snapshot[flattenedOffset + node],
        wasm._web_activation(layer, node),
        "the flattened snapshot contains the real current C activations",
      );
    }
    flattenedOffset += wasm._web_layer_size(layer);
  }
  const snapshotOutput = snapshot.subarray(snapshot.length - 10);
  const snapshotOutputSum = snapshotOutput.reduce((sum, value) => sum + value, 0);
  assert.ok(Math.abs(snapshotOutputSum - 1) < 1e-9);

  const syntheticCount = wasm._web_synthetic_samples();
  assert.equal(wasm._web_clear_synthetic_samples(), -1);
  assert.equal(wasm._web_add_synthetic_sample(3), -1);
  assert.equal(
    wasm._web_synthetic_samples(),
    syntheticCount,
    "training data cannot mutate during a partial shuffled epoch",
  );

  while (wasm._web_epochs_trained() === 0) {
    const processed = wasm._web_train_batch(137, 0.05);
    assert.ok(processed > 0 && processed <= 137);
  }
  assert.equal(wasm._web_epochs_trained(), 1);
  assert.equal(wasm._web_epoch_position(), 0);

  const batchedOutputs = Array.from({ length: 8 }, (_, sample) =>
    inspectOutputs(sample));

  assert.equal(
    saveModel(persistencePaths[0]),
    0,
    "Wasm saves a valid model at an epoch boundary",
  );
  const savedModelBytes = wasm.FS.readFile(persistencePaths[0]);
  const trailingModelBytes = new Uint8Array(savedModelBytes.length + 1);
  trailingModelBytes.set(savedModelBytes);
  trailingModelBytes[trailingModelBytes.length - 1] = 0xa5;
  wasm.FS.writeFile(persistencePaths[5], trailingModelBytes);
  wasm.FS.writeFile(
    persistencePaths[1],
    new TextEncoder().encode("not an enen model"),
  );
  wasm.FS.writeFile(
    persistencePaths[2],
    zeroModelBytes([63, 24, 16, 10]),
  );
  wasm.FS.writeFile(
    persistencePaths[3],
    modelHeaderBytes([64, 1000000, 4, 10]),
  );
  wasm.FS.writeFile(
    persistencePaths[4],
    modelHeaderBytes([64, 24, 16, 10]),
  );
  const preservedEpochs = wasm._web_epochs_trained();
  const preservedVersion = wasm._web_activation_version();
  const heapBytesBeforeOversizedLoad = wasm.HEAPF64.buffer.byteLength;
  assert.equal(loadModel(persistencePaths[3]), -1);
  assert.equal(
    wasm.HEAPF64.buffer.byteLength,
    heapBytesBeforeOversizedLoad,
    "an oversized serialized header is rejected without growing Wasm memory",
  );
  assert.equal(loadModel(persistencePaths[1]), -1);
  assert.equal(loadModel(persistencePaths[2]), -1);
  assert.equal(loadModel(persistencePaths[4]), -1);
  assert.equal(loadModel(persistencePaths[5]), -1);
  assert.deepEqual(modelShape(), [64, 24, 16, 10]);
  assert.equal(wasm._web_epochs_trained(), preservedEpochs);
  assert.equal(wasm._web_activation_version(), preservedVersion);
  for (let sample = 0; sample < batchedOutputs.length; sample += 1) {
    assert.deepEqual(
      inspectOutputs(sample),
      batchedOutputs[sample],
      "rejected Wasm loads preserve active model predictions exactly",
    );
  }

  const retainedTrainingSamples = wasm._web_training_samples();
  const retainedTestSamples = wasm._web_test_samples();
  const retainedSyntheticSamples = wasm._web_synthetic_samples();
  assert.equal(wasm._web_configure_model(32, 20, 99), 0);
  assert.equal(wasm._web_train_batch(5, 0.04), 5);
  assert.equal(wasm._web_evaluate(), 0);
  assert.ok(wasm._web_copy_test_sample(0) >= 0);
  assert.ok(wasm._web_predict() >= 0);
  assert.equal(
    loadModel(persistencePaths[0]),
    0,
    "Wasm loads a compatible saved model",
  );
  assert.deepEqual(modelShape(), [64, 24, 16, 10]);
  assert.equal(wasm._web_training_samples(), retainedTrainingSamples);
  assert.equal(wasm._web_test_samples(), retainedTestSamples);
  assert.equal(wasm._web_synthetic_samples(), retainedSyntheticSamples);
  assert.equal(wasm._web_epochs_trained(), 0);
  assert.equal(wasm._web_epoch_position(), 0);
  assert.equal(wasm._web_activation_version(), 0);
  assert.equal(wasm._web_last_training_label(), -1);
  assert.ok(wasm._web_activation_snapshot() > 0);
  assert.equal(wasm._web_activation_count(), 64 + 24 + 16 + 10);
  assert.equal(wasm._web_accuracy(), 0);
  assert.equal(wasm._web_loss(), 0);
  for (let digit = 0; digit < 10; digit += 1) {
    assert.equal(wasm._web_probability(digit), 0);
  }
  const loadedInputOffset =
    wasm._web_input_buffer() / Float64Array.BYTES_PER_ELEMENT;
  assert.ok(
    wasm.HEAPF64
      .subarray(loadedInputOffset, loadedInputOffset + 64)
      .every((pixel) => pixel === 0),
    "Wasm load resets the shared input buffer",
  );
  for (let sample = 0; sample < batchedOutputs.length; sample += 1) {
    assert.deepEqual(
      inspectOutputs(sample),
      batchedOutputs[sample],
      "Wasm model persistence round-trips predictions exactly",
    );
  }

  assert.equal(wasm._web_reset_model(11), 0);
  assert.deepEqual(modelShape(), [64, 24, 16, 10]);
  assert.equal(wasm._web_train_epoch(0.05), 0);
  for (let sample = 0; sample < batchedOutputs.length; sample += 1) {
    const oneCallOutputs = inspectOutputs(sample);
    for (let digit = 0; digit < oneCallOutputs.length; digit += 1) {
      assert.ok(
        Math.abs(oneCallOutputs[digit] - batchedOutputs[sample][digit]) < 1e-12,
        "batch boundaries do not change C training results",
      );
    }
  }

  assert.equal(wasm._web_configure_model(128, 64, 5), 0);
  assert.deepEqual(modelShape(), [64, 128, 64, 10]);

  for (let epoch = 0; epoch < 25; epoch += 1) {
    assert.equal(
      wasm._web_train_epoch(0.05),
      0,
      `Wasm C training completes epoch ${epoch + 1}`,
    );
  }
  assert.equal(wasm._web_epochs_trained(), 25);
  assert.equal(wasm._web_evaluate(), 0);
  const accuracy = wasm._web_accuracy();
  const loss = wasm._web_loss();
  assert.ok(accuracy >= 0.95, `Wasm test accuracy was ${accuracy}`);
  assert.ok(Number.isFinite(loss) && loss < 0.3, `Wasm test loss was ${loss}`);

  let robustCorrect = 0;
  for (let sample = 0; sample < wasm._web_test_samples(); sample += 1) {
    const label = wasm._web_copy_test_sample(sample);
    robustCorrect += wasm._web_predict() === label;
  }
  const robustAccuracy = robustCorrect / wasm._web_test_samples();
  assert.ok(
    robustAccuracy >= 0.94,
    `Wasm robust inference accuracy was ${robustAccuracy}`,
  );

  assert.ok(wasm._web_copy_test_sample(0) >= 0);
  assert.ok(wasm._web_predict() >= 0);
  let probabilitySum = 0;
  for (let digit = 0; digit < 10; digit += 1) {
    const probability = wasm._web_probability(digit);
    assert.ok(Number.isFinite(probability) && probability >= 0 && probability <= 1);
    probabilitySum += probability;
  }
  assert.ok(Math.abs(probabilitySum - 1) < 1e-9);

  console.log(
    `Wasm API accuracy: ${(accuracy * 100).toFixed(2)}%; ` +
    `robust inference: ${(robustAccuracy * 100).toFixed(2)}%`,
  );
} finally {
  for (const path of persistencePaths) {
    try {
      wasm.FS.unlink(path);
    } catch {
      // A test can fail before creating every persistence fixture.
    }
  }
  wasm._web_cleanup();
}

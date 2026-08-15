# UCI Digits Classifier in C

An educational multiclass classifier for the UCI Optical Recognition of
Handwritten Digits dataset. The implementation is plain C: no machine-learning
frameworks, hidden binary-classification path, or synthetic fallback dataset.

The model accepts 64 normalized pixels from an 8×8 image and predicts one of
ten classes (digits 0 through 9). Hidden layers use sigmoid activations; the
output layer uses softmax and is trained with cross-entropy loss.

## Build and verify

Requirements are a C11 compiler, GNU Make, and the standard math library.

```bash
make all
make check
```

Executables are written to `build/`. `make check` runs the core tests and also
reconverts the raw UCI files, byte-for-byte comparing the results with the
committed normalized CSV splits.

## Train and evaluate

```bash
./build/digits
```

This command:

1. Loads `conf/digits.conf`.
2. Trains a newly initialized network on `datasets/UCI_digits_train.csv`.
3. Reports accuracy, cross-entropy, and a confusion matrix on the held-out test
   split.
4. Saves the trained network to `models/digits.model`.

Evaluate a saved model without retraining:

```bash
./build/digits --load models/digits.model
```

The loader rejects malformed models and models whose input or output dimensions
do not match the configured dataset.

## Analysis programs

`accuracy` repeats the complete train/test experiment with independent random
initializations and summarizes the distribution of held-out accuracies:

```bash
./build/accuracy 10
./build/accuracy --load models/digits.model
```

`gym` compares learning rates and epoch counts. It reserves every fifth row of
the configured training split for validation so that the test split does not
influence hyperparameter selection. Each candidate starts from identical model
parameters and sees the same shuffle sequence.

```bash
./build/gym
./build/gym --load models/digits.model
```

The optional model is a starting point to clone and fine-tune; no temporary
model files are created.

## Dataset

The repository contains the official train/test organization in two forms:

- `datasets/optdigits.tra`: 3,823 original UCI training rows, with pixel
  intensities from 0 through 16.
- `datasets/optdigits.tes`: 1,797 original UCI test rows.
- `datasets/UCI_digits_train.csv`: normalized training rows used by the model.
- `datasets/UCI_digits_test.csv`: normalized test rows used by the model.

Each normalized row has exactly 65 comma-separated fields and no header:

```text
pixel_0,pixel_1,...,pixel_63,label
```

Pixels must be finite values in `[0, 1]`; labels must be integers from `0` to
`9`. The loader validates every field and converts each label to a ten-element
one-hot target. A malformed or empty dataset is an error rather than a partially
initialized training sample.

To regenerate normalized files from raw UCI data:

```bash
./build/convert_optdigits \
  datasets/optdigits.tra datasets/UCI_digits_train.csv \
  datasets/optdigits.tes datasets/UCI_digits_test.csv
```

The converter divides each original pixel intensity by 16 and preserves the
integer class label.

## Configuration

All programs read `conf/digits.conf`:

```ini
input_size=64
hidden_layers=128,64
output_size=10
learning_rate=0.05
epochs=25
train_dataset=datasets/UCI_digits_train.csv
test_dataset=datasets/UCI_digits_test.csv
```

The parser requires each field exactly once. Sizes and epoch counts must be
positive, the output must contain at least two classes, the learning rate must
be in `(0, 10]`, and both dataset paths must be nonempty.

## Implementation

The default architecture contains 17,226 trainable parameters:

```text
64 inputs -> 128 hidden -> 64 hidden -> 10 class probabilities
```

Weights use Xavier initialization. Training performs per-sample stochastic
gradient updates after shuffling the sample order each epoch. Softmax produces
one normalized distribution across all ten classes, prediction uses its argmax,
and cross-entropy measures how much probability was assigned to the true class.

The main source layout is:

```text
src/nn.c, src/nn.h                 network, training, metrics, persistence
src/dataset.c, src/dataset.h       validated normalized-CSV loading
src/config.c, src/config.h         strict configuration parsing
src/digits.c                       primary train/evaluate workflow
src/accuracy.c                     repeated-run statistics
src/gym.c                          validation-based hyperparameter comparison
src/convert_optdigits.c            raw UCI to normalized CSV conversion
tests/test_core.c                  dataset, softmax, training, and model tests
```

See [docs/DIGITS.md](docs/DIGITS.md) for the mathematical walkthrough and the
exact train/validation/test responsibilities.

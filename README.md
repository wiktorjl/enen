# Neural Network in C

An educational handwritten-digit classifier implemented in C without external
machine-learning libraries. It trains a configurable fully connected neural
network on 8×8 grayscale images and classifies digits from 0 through 9.

## Features

- Configurable fully connected network architecture
- One-hot multiclass targets and argmax classification
- Separate training and test datasets
- Xavier weight initialization and stochastic gradient descent
- OpenMP parallelization
- Model save/load support
- Hyperparameter and repeated-run accuracy tools

## Requirements

- GCC with OpenMP support
- GNU Make

## Build

```bash
make all
```

Executables are written to `build/`. To remove generated build artifacts:

```bash
make clean
```

## Quick start

Train the configured network and save it as `models/digits.model`:

```bash
./build/digits
```

Evaluate a saved model on the configured test split:

```bash
./build/digits --load models/digits.model
```

## Programs

### `digits`

The main training and evaluation program.

```bash
./build/digits
./build/digits --load models/digits.model
./build/digits --help
```

Without `--load`, it:

1. Loads `conf/digits.conf`.
2. Creates a new network.
3. Trains on `train_dataset`.
4. Evaluates on `test_dataset`.
5. Saves the trained model to `models/digits.model`.

### `gym`

Tests combinations of learning rate and epoch count. Every candidate trains on
the configured training split and reports MSE on the test split.

```bash
./build/gym
./build/gym --load models/digits.model
./build/gym --help
```

### `accuracy`

Trains multiple independently initialized networks and reports test accuracy,
mean, standard deviation, standard error, minimum, maximum, and a histogram.
It can also evaluate one saved model.

```bash
./build/accuracy 10
./build/accuracy --load models/digits.model
./build/accuracy --help
```

### Dataset utilities

```bash
./build/generate_digits
./build/convert_optdigits
```

`generate_digits` creates a small synthetic digit dataset.
`convert_optdigits` converts the original optical-digits files to the CSV format
used by the classifier.

## Configuration

The programs load `conf/digits.conf`:

```ini
input_size=64
hidden_layers=128,64
output_size=10
learning_rate=1.0
epochs=200
train_dataset=datasets/UCI_digits_train.csv
test_dataset=datasets/UCI_digits_test.csv
```

- `input_size`: number of input features; 64 represents an 8×8 image
- `hidden_layers`: comma-separated hidden-layer sizes
- `output_size`: number of digit classes
- `learning_rate`: stochastic-gradient-descent step size
- `epochs`: complete passes over the training split
- `train_dataset`: CSV used for parameter updates
- `test_dataset`: independent CSV used for evaluation

Each CSV row contains 64 normalized pixel values followed by a class label:

```text
pixel_0,pixel_1,...,pixel_63,label
```

The loader converts the label to a ten-element one-hot target vector.

## Project structure

```text
.
├── src/
│   ├── nn.c / nn.h             network creation, training, testing, persistence
│   ├── config.c / config.h     configuration parser
│   ├── tools.c / tools.h       data loading and numerical utilities
│   ├── digits.c                main classifier
│   ├── gym.c                   hyperparameter evaluation
│   ├── accuracy.c              repeated-run statistics
│   ├── generate_digits.c       synthetic dataset generator
│   └── convert_optdigits.c     dataset converter
├── conf/digits.conf
├── datasets/
├── docs/DIGITS.md
└── Makefile
```

## Network implementation

The default architecture is:

```text
64 inputs → 128 hidden → 64 hidden → 10 outputs
```

Forward propagation applies a sigmoid activation to each neuron. Training uses
one-hot targets, mean squared error, backpropagation, and per-sample stochastic
gradient updates. Samples are shuffled before each epoch. Classification selects
the output neuron with the largest activation.

Models use a compact binary serialization containing the architecture, weights,
and biases.

## Further reading

See [docs/DIGITS.md](docs/DIGITS.md) for a detailed walkthrough of the network,
training process, dataset, and model format.

## License

This project is intended for learning and teaching.

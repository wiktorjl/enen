# 8x8 Digit Recognition Neural Network in C

This project implements a feedforward neural network in plain C to classify handwritten-style `8x8` binary digit patterns (`0-9`).
No external ML libraries are used; training, inference, backpropagation, and model serialization are all implemented directly in this repository.

The repository also still contains legacy XOR tools (`xor`, `gym`, `accuracy`), but the primary workflow is now the `digits` program.

## What This Project Does

- Builds a fully connected neural network from `digits.conf`
- Loads a CSV dataset where each sample has:
  - `64` input features (flattened `8x8` pixels)
  - `1` class label (`0-9`)
- Converts labels to one-hot targets (length `10`)
- Trains with stochastic gradient descent + backpropagation
- Evaluates predictions with argmax classification
- Saves and loads models in a compact binary format (`digits.model`)

## Quick Start

### 1. Build

```bash
make generate_digits digits
```

### 2. Generate the dataset

```bash
./generate_digits
```

This creates `digits_dataset.csv` with 10 canonical `8x8` digit patterns.

### 3. Train a model

```bash
./digits
```

Default behavior:
- Loads `digits.conf`
- Loads `digits_dataset.csv`
- Trains a new model
- Prints MSE + per-sample predictions
- Saves `digits.model`

### 4. Load and test an existing model

```bash
./digits --load digits.model
```

## Configuration

Primary config file: `digits.conf`

```ini
input_size=64
hidden_layers=128,64
output_size=10
learning_rate=0.5
epochs=50000
dataset=digits_dataset.csv
```

Meaning of each field:
- `input_size`: number of input neurons (`8x8 = 64`)
- `hidden_layers`: comma-separated hidden layer widths
- `output_size`: number of classes (`10` digits)
- `learning_rate`: SGD step size
- `epochs`: full passes through the dataset
- `dataset`: path to CSV data file

## How The Neural Network Works (Using This Project)

This section explains the actual training path in this repo (`digits.c`, `nn.c`, `tools.c`).

### 1. Data representation

Each row in `digits_dataset.csv` is:

`pixel_0, pixel_1, ..., pixel_63, label`

- Pixels are `0.0` or `1.0`
- Label is an integer `0-9`

During load (`load_dataset_multiclass` in `tools.c`), labels are converted to one-hot vectors.

Example label `5` becomes:

`[0,0,0,0,0,1,0,0,0,0]`

### 2. Network architecture

From `digits.conf`, default architecture is:

`64 -> 128 -> 64 -> 10`

- Layer 0: input pixels
- Layer 1: hidden features
- Layer 2: hidden features
- Layer 3: class score outputs (one neuron per digit class)

Parameter count with this default:
- Weights: `64*128 + 128*64 + 64*10 = 17024`
- Biases: `128 + 64 + 10 = 202`
- Total trainable parameters: `17226`

### 3. Weight storage and indexing

Weights are stored as flattened row-major arrays per layer transition:

`weights[layer][i * next_size + j]`

Interpretation:
- `i` = neuron index in current layer
- `j` = neuron index in next layer

So the connection from neuron `i` to neuron `j` is one contiguous lookup, not a 2D allocation.

### 4. Forward pass

Implemented in `forward_pass` (`nn.c`):

For each neuron `j` in the next layer:

`z_j = bias_j + sum_i (a_i * w_ij)`

`a_j = sigmoid(z_j)`

Where:
- `a_i` is activation from previous layer
- `w_ij` is weight from neuron `i` to `j`
- `sigmoid(x) = 1 / (1 + exp(-x))`

This repeats layer-by-layer until the output vector of length `10` is produced.

### 5. Prediction rule

After the forward pass, classification uses argmax:

- Find the output neuron with highest activation
- Its index is the predicted digit

In `test_nn_multiclass` (`nn.c`), this is exactly how `predicted` is computed.

### 6. Loss signal and error terms

For training, each output neuron gets an error term:

`error_k = target_k - output_k`

`delta_k = error_k * sigmoid_derivative(output_k)`

`sigmoid_derivative(s) = s * (1 - s)`

This gives both direction and scale for correction at the output layer.

### 7. Backpropagation through hidden layers

Implemented in `backward_pass` (`nn.c`):

For each hidden neuron `i`:

`delta_i = (sum_j delta_j_next * w_ij) * sigmoid_derivative(a_i)`

Intuition:
- If downstream neurons are wrong, error flows backward
- A hidden neuron gets more blame if it strongly contributed via large weights
- Derivative term reduces updates when the neuron is saturated

### 8. Parameter update rule (SGD)

For each connection `i -> j`:

`w_ij += learning_rate * delta_j * a_i`

For each bias `j`:

`b_j += learning_rate * delta_j`

This runs per sample (online SGD style) inside each epoch.

### 9. Training loop in this project

`train_nn_multiclass` (`nn.c`) does:

1. Build a shuffled index order for samples
2. For each sample in that order:
   - forward pass
   - backward pass
   - immediate weight update
3. Repeat for `epochs`

Why shuffle matters:
- Prevents learning artifacts from fixed ordering
- Usually improves stability of SGD updates

### 10. MSE metric and classification accuracy

The project reports:
- Mean squared error via `test_nn_and_get_mse_multiclass`
- Classification accuracy via `test_nn_multiclass`

The MSE function sums squared error across all 10 outputs per sample and divides by number of samples.

### 11. Why this learns digits here

In this repo, the dataset contains one clean prototype per digit.
That makes the mapping simple enough for a small dense network, so the model typically converges to perfect training accuracy.

For noisy real-world handwriting, you would typically need:
- many more samples
- train/validation/test splits
- better normalization/augmentation
- potentially different activations/losses (for example, softmax + cross-entropy)

## Model Format (`digits.model`)

Serialization is implemented in `save_net` / `load_net` (`nn.c`).

Binary file layout:
1. Magic number (`ENEN`)
2. Version
3. Number of layers
4. Layer sizes array
5. All weight arrays
6. All bias arrays

This allows exact architecture + parameter restoration for inference or continued training.

## OpenMP Parallelization

If built with OpenMP (`-fopenmp`), forward/backward loops over neurons run in parallel (`#pragma omp parallel for`) in `nn.c`.
This speeds layer computations on multi-core CPUs without changing model behavior.

## Available Programs

Primary:
- `digits`: train/test digit classifier, save/load model
- `generate_digits`: create `digits_dataset.csv`

Legacy XOR tools:
- `xor`: train XOR network using `xornet.conf`
- `gym`: XOR hyperparameter sweep
- `accuracy`: repeated XOR runs with summary stats

You can build everything with:

```bash
make all
```

## Project Files

- `digits.c`: digit workflow entry point
- `generate_digits.c`: dataset generator (`8x8` digit patterns)
- `nn.c`, `nn.h`: neural network core (forward, backward, train, test, save/load)
- `tools.c`, `tools.h`: math and dataset utilities
- `config.c`, `config.h`: config parser
- `digits.conf`: digit model configuration
- `xornet.conf`: legacy XOR configuration
- `DIGITS.md`: expanded implementation walkthrough

## License

Educational project for learning neural network implementation details in C.

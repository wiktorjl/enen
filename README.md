# Neural Network in C

A clean, educational neural network implementation in C that demonstrates fundamental concepts through two examples: the classic XOR problem and 8×8 handwritten digit recognition. Built from scratch with no external ML libraries, this project shows how neural networks work at a low level using backpropagation and gradient descent.

## Features

- **Pure C implementation** - No external ML libraries, educational and transparent
- **Dynamic architecture** - Configure any network size via config files
- **Multi-class classification** - Supports binary and multi-class problems
- **Model persistence** - Save and load trained models
- **OpenMP parallelization** - Efficient training on multi-core systems
- **Comprehensive tooling** - Training, testing, hyperparameter tuning, and statistical analysis

## Table of Contents

- [Building the Project](#building-the-project)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Available Programs](#available-programs)
- [Configuration](#configuration)
- [How the Neural Network Works](#how-the-neural-network-works)
- [Deep Dive: DIGITS.md](#deep-dive)

## Building the Project

Requirements: GCC with OpenMP support, GNU Make

```bash
make all    # Build all programs
make clean  # Remove build artifacts
```

All executables are placed in `build/`, and trained models are saved to `models/`.

## Project Structure

```
.
├── src/                # Source code
│   ├── nn.c/nn.h           # Core neural network implementation
│   ├── config.c/config.h   # Configuration file parser
│   ├── tools.c/tools.h     # Utilities and data loading
│   ├── xor.c               # XOR training program
│   ├── digits.c            # 8×8 digit recognition program
│   ├── generate_digits.c   # Dataset generator for digits
│   ├── gym.c               # Hyperparameter tuning tool
│   └── accuracy.c          # Statistical analysis tool
├── conf/               # Configuration files
│   ├── xornet.conf         # XOR network configuration
│   └── digits.conf         # Digit network configuration
├── build/              # Compiled executables and object files
├── models/             # Saved neural network models
├── docs/               # Documentation
│   └── DIGITS.md       # In-depth neural network walkthrough
└── Makefile            # Build system
```

## Quick Start

### Train the XOR Network

```bash
make xor
./build/xor
```

Output shows the learned XOR function:
```
=== What did we learn? ===
[0,0] → 0.043 (want 0)
[0,1] → 0.967 (want 1)
[1,0] → 0.982 (want 1)
[1,1] → 0.021 (want 0)
```

### Train the Digit Classifier

```bash
make digits generate_digits
./build/generate_digits    # Create training data
./build/digits              # Train network

# Test saved model
./build/digits --load models/digits.model
```

## Available Programs

### 1. `xor` - XOR Function Learning

Trains a neural network to learn the XOR function, demonstrating that neural networks can solve non-linearly separable problems.

**Usage:**
```bash
./build/xor
```

**Features:**
- Trains on 4 XOR examples: (0,0)→0, (0,1)→1, (1,0)→1, (1,1)→0
- Configurable architecture via `xornet.conf`
- Saves trained model to `models/xor.model`
- Displays final predictions and Mean Squared Error

### 2. `digits` - Handwritten Digit Recognition

Classifies 8×8 pixel handwritten digits (0-9) using a multi-layer neural network.

**Usage:**
```bash
./build/digits                              # Train new model
./build/digits --load models/digits.model   # Load trained model
./build/digits --help                       # Show help
```

**Features:**
- 64 input neurons (8×8 pixels)
- Two hidden layers (128 and 64 neurons by default)
- 10 output neurons (one per digit)
- ~17,000 trainable parameters
- Multi-threaded training with OpenMP
- Model serialization for instant testing

**Sample Output:**
```
Network Architecture: 64-128-64-10
Using 16 OpenMP threads
Training network for 50000 epochs...
Final MSE: 0.000008

Sample 0: predicted=0, expected=0 [OK]
Sample 1: predicted=1, expected=1 [OK]
...
Accuracy: 10/10 (100.0%)
```

### 3. `generate_digits` - Training Data Generator

Creates the 8×8 pixel representations of digits 0-9 for training.

**Usage:**
```bash
./build/generate_digits
```

Generates `digits_dataset.csv` with hardcoded pixel patterns for each digit.

### 4. `gym` - Hyperparameter Tuning

Systematically tests different learning rates and epoch counts to find optimal training parameters.

**Usage:**
```bash
./build/gym                               # Test from scratch
./build/gym --load models/xor.model       # Fine-tune existing model
./build/gym --help                        # Show help
```

**Test Matrix:**
- Learning rates: 0.01, 0.1, 0.5, 1.0
- Epochs: 10,000, 25,000, 50,000, 100,000

**Output:**
```
| Learning Rate | Rounds | MSE      |
|---------------|--------|----------|
| 0.01          | 10000  | 0.234567 |
| 0.50          | 50000  | 0.000123 |
...
```

### 5. `accuracy` - Statistical Analysis

Evaluates training consistency by running multiple independent training sessions with different random initializations.

**Usage:**
```bash
./build/accuracy 100                         # Train 100 times
./build/accuracy --load models/xor.model     # Test saved model
./build/accuracy --help                      # Show help
```

**Provides:**
- Average accuracy across all runs
- Standard deviation and standard error
- Min/max accuracy
- ASCII histogram of accuracy distribution

**Sample Output:**
```
Run   50/100  | Acc: 100.0% | Avg:  94.5% [#####################################]

--- Statistical Summary over 100 runs ---
Average Accuracy:  94.50%
Standard Deviation: 12.34%
Standard Error:     1.23%
Minimum Accuracy:  50.00%
Maximum Accuracy: 100.00%
```

## Configuration

The network configuration is stored in `xornet.conf`:

```
input_size=2
hidden_layers=3,4,3
output_size=1
learning_rate=0.5
epochs=10000
dataset=xor_dataset.csv
```

**Configuration parameters:**
- `input_size`: Number of input neurons (2 for XOR, 64 for digits)
- `hidden_layers`: Comma-separated layer sizes (e.g., `3,4,3` for three hidden layers)
- `output_size`: Number of output neurons (1 for XOR, 10 for digits)
- `learning_rate`: Step size for gradient descent (typical: 0.1-0.5)
- `epochs`: Number of training iterations (10,000-100,000)
- `dataset`: Path to CSV file containing training data

**Example configurations:**

`conf/xornet.conf`:
```
input_size=2
hidden_layers=3,4,3
output_size=1
learning_rate=0.5
epochs=10000
dataset=xor_dataset.csv
```

`conf/digits.conf`:
```
input_size=64
hidden_layers=128,64
output_size=10
learning_rate=0.5
epochs=50000
dataset=digits_dataset.csv
```

## How the Neural Network Works

### The XOR Problem

The XOR (exclusive OR) function is a classic problem in neural network research because it's not linearly separable. This means you cannot draw a single straight line to separate the true outputs from false outputs.

**XOR Truth Table:**
```
Input A | Input B | Output
--------|---------|--------
   0    |    0    |   0
   0    |    1    |   1
   1    |    0    |   1
   1    |    1    |   0
```

XOR requires at least one hidden layer to learn, making it an ideal benchmark for testing neural network implementations.

### Network Architecture

The implementation supports **fully dynamic architectures** configured via config files.

**XOR Example (2-3-4-3-1 architecture):**
```
Input Layer (2)  →  Hidden (3)  →  Hidden (4)  →  Hidden (3)  →  Output (1)
      [x]               [h₁]          [h₁]          [h₁]            [y]
      [y]               [h₂]          [h₂]          [h₂]
                        [h₃]          [h₃]          [h₃]
                                      [h₄]
```

**Digits Example (64-128-64-10 architecture):**
```
Input (64 pixels)  →  Hidden (128)  →  Hidden (64)  →  Output (10 classes)
```

**Components:**
- **Weights**: Matrices connecting each layer pair (dynamically allocated)
- **Biases**: One bias value per neuron in hidden and output layers
- **Activations**: Cached outputs from each layer during forward pass

### Forward Propagation

Forward propagation computes the network's output by passing data through each layer sequentially.

**Algorithm (nn.c:75-90):**

For each layer transition:
1. **Weighted Sum**: Sum all inputs × their weights, plus bias
2. **Activation**: Apply sigmoid function to the sum
3. **Pass Forward**: Use result as input to next layer

**Mathematical formulation:**
```
For neuron j in layer L+1:
    z_j = bias_j + Σ(activation_i × weight_ij)  // sum over layer L
    activation_j = sigmoid(z_j) = 1 / (1 + e^(-z_j))
```

**Code Implementation:**
```c
for (int layer = 0; layer < num_layers - 1; layer++) {
    for (int j = 0; j < next_size; j++) {
        double sum = biases[layer][j];
        for (int i = 0; i < current_size; i++) {
            sum += activations[layer][i] * weights[layer][i * next_size + j];
        }
        activations[layer + 1][j] = sigmoid(sum);
    }
}
```

### Backpropagation

Backpropagation adjusts weights to minimize prediction error by propagating error gradients backward through the network.

**Three Phases:**

**1. Output Layer Error (nn.c:101-106)**
```
For each output neuron:
    error = expected - actual
    delta = error × sigmoid'(activation)
```

**2. Hidden Layer Error Propagation (nn.c:108-119)**
```
For each hidden layer (backward):
    For each neuron:
        error = Σ(next_layer_delta × connecting_weight)
        delta = error × sigmoid'(activation)
```

**3. Weight Updates (nn.c:122-136)**
```
For each weight:
    weight += learning_rate × next_neuron_delta × current_activation
For each bias:
    bias += learning_rate × neuron_delta
```

**Sigmoid Derivative:**
```c
sigmoid'(x) = sigmoid(x) × (1 - sigmoid(x))
```

This convenient form allows computing the derivative directly from the forward pass output.

**Key Insight:** Weights are adjusted proportionally to:
- **Error gradient** (how wrong the prediction was)
- **Learning rate** (step size for updates)
- **Input activation** (how much this neuron contributed)

### Training Process

**Stochastic Gradient Descent with Shuffling (nn.c:145-157, nn.c:318-330)**

```
For each epoch:
    1. Shuffle training examples
    2. For each example:
        a. Forward pass → compute prediction
        b. Backward pass → compute gradients
        c. Update weights immediately
```

**Why Shuffle?**
Random order each epoch prevents the network from:
- Memorizing the sequence
- Overfitting to the order
- Getting stuck in poor local minima

**Convergence:**
After sufficient epochs, weights stabilize and the network learns to map inputs to correct outputs. Typical convergence:
- **XOR**: ~5,000-10,000 epochs
- **Digits**: ~30,000-50,000 epochs

**Parallelization:**
The implementation uses OpenMP to parallelize neuron computations within each layer, significantly speeding up training on multi-core systems.

### Activation Function

The **sigmoid function** (tools.c:76-78) is used as the activation function:

```
σ(x) = 1 / (1 + e^(-x))
```

**Properties:**
- **Range:** (0, 1) - outputs are always between 0 and 1
- **Non-linear:** Enables learning of complex patterns
- **Differentiable:** Required for backpropagation
- **Smooth:** Small changes in input cause small changes in output

**Sigmoid derivative** (tools.c:80-82):
```
σ'(x) = σ(x) × (1 - σ(x))
```

This derivative is used during backpropagation to calculate gradients. The convenient form means we can compute it directly from the sigmoid output without storing the original input.

**Why sigmoid for XOR?**
- Squashes outputs to (0,1) range, matching our binary target values
- Smooth gradients help convergence
- Historical choice for binary classification problems

---

## Deep Dive

For a comprehensive, blog-post quality walkthrough of how neural networks work using the digits classifier as an example, see:

**[docs/DIGITS.md](docs/DIGITS.md)** - A complete primer on neural networks for software engineers

This document covers:
- Network architecture and data structures
- Forward and backward propagation with detailed examples
- Training process and convergence
- Model serialization
- Multi-threading with OpenMP
- Mathematical foundations
- Complete code reference guide

Perfect for understanding neural networks from first principles!

## Performance

**Training Speed (50,000 epochs, 10 samples):**
- Single-threaded: ~30-40 seconds
- Multi-threaded (16 cores): ~5-8 seconds

**Accuracy:**
- XOR: 100% (4/4 samples)
- Digits: 100% (10/10 samples) on training set

**Model Size:**
- XOR model: ~1 KB
- Digits model: ~135 KB

## License

This is an educational project demonstrating neural network fundamentals. Use freely for learning and teaching.

# XOR Neural Network in C

A simple feedforward neural network implementation in C that learns the XOR function using backpropagation. This project demonstrates fundamental neural network concepts including forward propagation, backpropagation, and gradient descent optimization.

## Table of Contents

- [Building the Project](#building-the-project)
- [Available Programs](#available-programs)
- [Configuration](#configuration)
- [How the Neural Network Works](#how-the-neural-network-works)
  - [The XOR Problem](#the-xor-problem)
  - [Network Architecture](#network-architecture)
  - [Forward Propagation](#forward-propagation)
  - [Backpropagation](#backpropagation)
  - [Training Process](#training-process)
  - [Activation Function](#activation-function)

## Building the Project

The project uses a Makefile for compilation. Ensure you have GCC and the math library installed.

### Build all programs:
```bash
make all
```

### Build individual programs:
```bash
make xor       # Main training and testing program
make gym       # Hyperparameter tuning utility
make accuracy  # Statistical accuracy analysis
```

### Clean build artifacts:
```bash
make clean
```

## Available Programs

### 1. `xor` - Main Neural Network Training

The primary program that trains a neural network on the XOR problem and displays the results.

**Usage:**
```bash
./xor
```

**What it does:**
- Loads configuration from `xornet.conf`
- Initializes the network with random weights
- Loads training data from the configured dataset file
- Trains the network using backpropagation
- Tests and displays the learned XOR function outputs

**Example output:**
```
=== What did we learn? ===
[0,0] → 0.043 (want 0)
[1,1] → 0.021 (want 0)
[0,1] → 0.967 (want 1)
[1,0] → 0.982 (want 1)
```

### 2. `gym` - Hyperparameter Tuning

A utility that tests different combinations of learning rates and training rounds to find optimal hyperparameters.

**Usage:**
```bash
./gym
```

**What it does:**
- Tests learning rates: 0.01, 0.1, 0.5, 1.0
- Tests training rounds: 1000, 5000, 10000, 20000
- Calculates Mean Squared Error (MSE) for each combination
- Outputs results in a formatted table

**Example output:**
```
| Learning Rate | Rounds | MSE      |
|---------------|--------|----------|
| 0.01          | 1000   | 0.234567 |
| 0.01          | 5000   | 0.098765 |
...
```

### 3. `accuracy` - Statistical Analysis

Runs multiple training sessions with random weight initializations to analyze training consistency and success rate.

**Usage:**
```bash
./accuracy <number_of_runs>
```

**Example:**
```bash
./accuracy 100  # Run 100 training sessions
```

**What it does:**
- Runs N independent training sessions
- Tracks accuracy for each run (0%, 25%, 50%, 75%, or 100%)
- Displays live progress with running average
- Calculates statistical measures:
  - Average accuracy
  - Standard deviation
  - Standard error
  - Min/Max accuracy
- Shows ASCII histogram of accuracy distribution

**Example output:**
```
Run    1/100  | Acc: 100.0% | Avg: 100.0% [########################################]
Run    2/100  | Acc:  75.0% | Avg:  87.5% [###################################     ]
...

--- Statistical Summary over 100 runs ---
Average Accuracy:  94.50%
Standard Deviation: 12.34%
Standard Error:     1.23%
Minimum Accuracy:  50.00%
Maximum Accuracy: 100.00%

--- Accuracy Distribution Histogram ---
  0.0% | [          ] (0 runs)
 25.0% | [####      ] (2 runs)
 50.0% | [########  ] (4 runs)
 75.0% | [############] (6 runs)
100.0% | [##################################################] (88 runs)
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
- `input_size`: Number of input neurons (2 for XOR: x and y)
- `hidden_layers`: Sizes of hidden layers (currently supports only 2 hidden neurons in code)
- `output_size`: Number of output neurons (1 for XOR)
- `learning_rate`: Step size for gradient descent (default: 0.5)
- `epochs`: Number of training iterations (default: 10000)
- `dataset`: Path to CSV file containing training data

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

This implementation uses a **2-2-1 architecture**:

```
Input Layer (2 neurons)  →  Hidden Layer (2 neurons)  →  Output Layer (1 neuron)
      [x]                          [h₁]                        [y]
      [y]                          [h₂]
```

**Components:**
- **Weights (input→hidden)**: `weights_ih[2][2]` - 4 weights connecting input to hidden layer
- **Weights (hidden→output)**: `weights_ho[2][1]` - 2 weights connecting hidden to output layer
- **Biases (hidden)**: `bias_hidden[2]` - 2 bias values for hidden neurons
- **Bias (output)**: `bias_output[1]` - 1 bias value for output neuron

### Forward Propagation

Forward propagation computes the network's output for a given input by passing data through each layer.

**Mathematical formulation:**

1. **Hidden layer calculation** (nn.c:12-13):
```
h₁ = sigmoid(x·w_ih[0][0] + y·w_ih[0][1] + bias_h[0])
h₂ = sigmoid(x·w_ih[1][0] + y·w_ih[1][1] + bias_h[1])
```

2. **Output layer calculation** (nn.c:16):
```
output = sigmoid(h₁·w_ho[0][0] + h₂·w_ho[1][0] + bias_o[0])
```

Each neuron:
1. Computes weighted sum of inputs plus bias
2. Applies sigmoid activation function
3. Passes result to next layer

### Backpropagation

Backpropagation is the learning algorithm that adjusts weights to minimize prediction error. It works backward through the network, computing gradients using the chain rule of calculus.

**Algorithm steps:**

1. **Calculate output error** (nn.c:20):
```
error_output = (expected - actual) × sigmoid'(output)
```

2. **Update output layer weights** (nn.c:22-24):
```
w_ho[i] += learning_rate × error_output × h[i]
bias_output += learning_rate × error_output
```

3. **Calculate hidden layer errors** (nn.c:26-29):
```
error_hidden[j] = error_output × w_ho[j] × sigmoid'(h[j])
```

4. **Update hidden layer weights** (nn.c:32-37):
```
w_ih[i][j] += learning_rate × error_hidden[i] × input[j]
bias_hidden[i] += learning_rate × error_hidden[i]
```

**Key insight:** Each weight is adjusted proportionally to:
- How much it contributed to the error (gradient)
- The learning rate (step size)
- The activation of the neuron feeding into it

### Training Process

The training loop (nn.c:42-63) follows this procedure:

1. **For each epoch:**
   - Shuffle input order to prevent bias
   - For each training example:
     - Perform forward pass to get prediction
     - Perform backward pass to update weights

2. **Input shuffling:** Random order each epoch prevents the network from memorizing sequence patterns

3. **Convergence:** After enough iterations, weights stabilize and the network learns the XOR mapping

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

## Project Structure

```
.
├── xor.c              # Main program entry point
├── nn.c               # Neural network forward/backward pass logic
├── nn.h               # Neural network function declarations
├── tools.c            # Utility functions (sigmoid, initialization, data loading)
├── tools.h            # Utility function declarations and Net struct definition
├── config.c           # Configuration file parsing
├── config.h           # Configuration struct definition
├── gym.c              # Hyperparameter tuning utility
├── accuracy.c         # Statistical accuracy analysis tool
├── Makefile           # Build system configuration
├── xornet.conf        # Network configuration file
└── xor_dataset.csv    # Training data (4 XOR examples)
```

## License

This is an educational project demonstrating basic neural network concepts.

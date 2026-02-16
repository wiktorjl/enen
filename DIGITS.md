# DIGITS.md - Neural Network Walkthrough

## Table of Contents
1. [Overview](#overview)
2. [Network Architecture](#network-architecture)
3. [Data Structures](#data-structures)
4. [Data Flow](#data-flow)
5. [Forward Propagation](#forward-propagation)
6. [Backward Propagation](#backward-propagation)
7. [Training Process](#training-process)
8. [Model Serialization](#model-serialization)
9. [Multi-Threading](#multi-threading)
10. [Code Reference Guide](#code-reference-guide)

---

## Overview

This neural network implementation performs 8×8 pixel handwritten digit recognition (0-9). The system is built in pure C with no external ML libraries, using:

- **Fully connected** (dense) layers
- **Sigmoid** activation function
- **Backpropagation** with stochastic gradient descent
- **One-hot encoding** for multi-class classification
- **OpenMP** parallelization for performance

**Key Files:**
- `nn.c/nn.h` - Core neural network engine
- `digits.c` - Digit classification program
- `tools.c/tools.h` - Dataset loading and utilities
- `config.c/config.h` - Configuration file parser

---

## Network Architecture

### Default Configuration (digits.conf)

```
input_size=64        # 8×8 pixels = 64 inputs
hidden_layers=128,64 # Two hidden layers: 128 neurons, then 64 neurons
output_size=10       # 10 classes (digits 0-9)
learning_rate=0.5
epochs=50000
```

### Layer Structure: 64 → 128 → 64 → 10

```
Input Layer (64)         Hidden Layer 1 (128)      Hidden Layer 2 (64)       Output Layer (10)
┌──────────┐             ┌──────────┐              ┌──────────┐              ┌──────────┐
│ pixel 0  │────┐        │ neuron 0 │──┐           │ neuron 0 │──┐           │ digit 0  │
│ pixel 1  │────┼────────│ neuron 1 │──┼───────────│ neuron 1 │──┼───────────│ digit 1  │
│   ...    │────┘        │   ...    │──┘           │   ...    │──┘           │   ...    │
│ pixel 63 │             │ neuron127│              │ neuron 63│              │ digit 9  │
└──────────┘             └──────────┘              └──────────┘              └──────────┘
```

**Total Parameters:**
- Weights: (64×128) + (128×64) + (64×10) = 8,192 + 8,192 + 640 = **17,024 weights**
- Biases: 128 + 64 + 10 = **202 biases**
- **Total: 17,226 trainable parameters**

---

## Data Structures

### Net Structure (nn.h:6-13)

```c
typedef struct NetStruct {
    int num_layers;          // Total layers including input (e.g., 4)
    int *layer_sizes;        // Array: [64, 128, 64, 10]

    double **weights;        // 3 weight matrices (num_layers - 1)
    double **biases;         // 3 bias vectors
    double **activations;    // 4 activation vectors (cached during forward pass)
} Net;
```

**Memory Layout:**

```
weights[0]: 64×128 matrix  (flattened to 1D array of 8,192 doubles)
weights[1]: 128×64 matrix  (flattened to 1D array of 8,192 doubles)
weights[2]: 64×10 matrix   (flattened to 1D array of 640 doubles)

biases[0]: vector of 128 doubles
biases[1]: vector of 64 doubles
biases[2]: vector of 10 doubles

activations[0]: input layer (64 doubles)
activations[1]: hidden layer 1 (128 doubles)
activations[2]: hidden layer 2 (64 doubles)
activations[3]: output layer (10 doubles)
```

**Weight Matrix Indexing (row-major order):**

```c
// To access weight from neuron i in layer L to neuron j in layer L+1:
weight = net->weights[layer][i * next_size + j];

// Example: Weight from input neuron 5 to hidden neuron 10
weight = net->weights[0][5 * 128 + 10];
```

---

## Data Flow

### Dataset Format (digits_dataset.csv)

```csv
0.0,1.0,1.0,...,0.0,0    # 64 pixel values (0.0 or 1.0) + label (0-9)
0.0,0.0,0.0,...,1.0,1    # 10 rows total (one per digit)
...
```

### One-Hot Encoding (tools.c:126-174)

**Input:** Label = 5

**Output Array:**
```
[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
  0    1    2    3    4    5    6    7    8    9
```

**Code (tools.c:157-164):**
```c
// Read label and convert to one-hot encoding
if (token) {
    int label = atoi(token);
    if (label >= 0 && label < num_classes) {
        expected[count][label] = 1.0;  // Set only the target class to 1.0
    }
}
```

### Complete Data Flow

```
CSV File                  One-Hot Encoding           Network Input/Output
┌─────────────┐          ┌─────────────┐           ┌─────────────┐
│ 0,1,1,...,5 │──parse──→│ inputs[64]  │──────────→│             │
│             │          │ expected[10]│           │   NETWORK   │
│             │          │ [0,0,0,0,0, │           │             │
│             │          │  1,0,0,0,0] │←──────────│             │
└─────────────┘          └─────────────┘           └─────────────┘
                                                    activations[3][10]
```

---

## Forward Propagation

### Algorithm (nn.c:73-86)

For each layer transition L → L+1:

1. **Initialize:** Start with bias
2. **Weighted Sum:** Add contribution from each input neuron
3. **Activate:** Apply sigmoid function

**Mathematical Formula:**

```
For neuron j in layer L+1:

z_j = bias[j] + Σ(activation[i] × weight[i→j])  // i over all neurons in layer L
activation[j] = sigmoid(z_j) = 1 / (1 + e^(-z_j))
```

### Code Implementation (nn.c:73-86)

```c
void forward_pass(const double *inputs, Net *net) {
    // Copy input pixels to first layer
    memcpy(net->activations[0], inputs, sizeof(double) * net->layer_sizes[0]);

    // Propagate through each layer
    for (int layer = 0; layer < net->num_layers - 1; layer++) {
        int current_size = net->layer_sizes[layer];
        int next_size = net->layer_sizes[layer + 1];

        // Compute activation for each neuron in next layer
        #pragma omp parallel for schedule(static)  // Parallel across neurons
        for (int j = 0; j < next_size; j++) {
            double sum = net->biases[layer][j];  // Start with bias

            // Add weighted inputs from previous layer
            for (int i = 0; i < current_size; i++) {
                sum += net->activations[layer][i] * net->weights[layer][i * next_size + j];
            }

            net->activations[layer + 1][j] = sigmoid(sum);  // Apply activation
        }
    }
}
```

### Example: Input → Hidden Layer 1

```
Input: [0.0, 1.0, 1.0, ..., 0.0]  (64 pixels)
                ↓
For hidden neuron 0:
    sum = bias[0][0]
    sum += 0.0 × weight[0][0×128 + 0]    // pixel 0
    sum += 1.0 × weight[0][1×128 + 0]    // pixel 1
    sum += 1.0 × weight[0][2×128 + 0]    // pixel 2
    ...
    activation[1][0] = sigmoid(sum)
```

### Sigmoid Activation (tools.c:60-62)

```c
double sigmoid(double input) {
    return 1.0 / (1.0 + exp(-input));
}
```

**Properties:**
- Maps (-∞, +∞) to (0, 1)
- Smooth, differentiable
- Output can be interpreted as probability

---

## Backward Propagation

### Overview

Backpropagation computes how much each weight contributed to the error and updates weights to reduce error.

**Three Phases:**
1. **Output Layer Error:** Compare prediction to target
2. **Hidden Layer Error:** Propagate error backward
3. **Weight Update:** Adjust weights proportional to error

### Phase 1: Output Layer Error (nn.c:103-107)

**Formula:**
```
For output neuron i:
    error = expected[i] - actual[i]
    delta[i] = error × sigmoid'(activation[i])
```

**Sigmoid Derivative (tools.c:64-66):**
```c
double sigmoid_derivative(double sigmoid_output) {
    return sigmoid_output * (1.0 - sigmoid_output);
}
```

**Code (nn.c:103-107):**
```c
int output_layer = num_layers - 1;
#pragma omp parallel for schedule(static)
for (int i = 0; i < net->layer_sizes[output_layer]; i++) {
    double error = expected[i] - net->activations[output_layer][i];
    deltas[output_layer][i] = error * sigmoid_derivative(net->activations[output_layer][i]);
}
```

**Example (digit 5, network output slightly wrong):**
```
Expected: [0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0]
Actual:   [0.01, 0.02, 0.01, 0.03, 0.05, 0.85, 0.01, 0.01, 0.01, 0.00]

Errors:   [-0.01, -0.02, -0.01, -0.03, -0.05, +0.15, -0.01, -0.01, -0.01, 0.00]
                                              ^^^^
                                        Positive error for correct class
```

### Phase 2: Hidden Layer Error (nn.c:109-120)

**Formula:**
```
For hidden neuron i in layer L:
    error[i] = Σ(delta[j] × weight[i→j])  // j over all neurons in layer L+1
    delta[i] = error[i] × sigmoid'(activation[i])
```

**Code (nn.c:109-120):**
```c
for (int layer = num_layers - 2; layer >= 0; layer--) {  // Backward
    int current_size = net->layer_sizes[layer];
    int next_size = net->layer_sizes[layer + 1];

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < current_size; i++) {
        double error = 0.0;

        // Accumulate error from next layer
        for (int j = 0; j < next_size; j++) {
            error += deltas[layer + 1][j] * net->weights[layer][i * next_size + j];
        }

        deltas[layer][i] = error * sigmoid_derivative(net->activations[layer][i]);
    }
}
```

### Phase 3: Weight Update (nn.c:122-135)

**Formula (Gradient Descent):**
```
weight[i→j] += learning_rate × delta[j] × activation[i]
bias[j] += learning_rate × delta[j]
```

**Intuition:**
- If `delta[j]` is positive: neuron j needs higher activation → increase weights
- If `activation[i]` is large: neuron i strongly influences j → update more
- `learning_rate` controls step size (e.g., 0.5)

**Code (nn.c:122-135):**
```c
for (int layer = 0; layer < num_layers - 1; layer++) {
    int current_size = net->layer_sizes[layer];
    int next_size = net->layer_sizes[layer + 1];

    // Update weights
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < current_size; i++) {
        for (int j = 0; j < next_size; j++) {
            net->weights[layer][i * next_size + j] +=
                learning_rate * deltas[layer + 1][j] * net->activations[layer][i];
        }
    }

    // Update biases
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < next_size; j++) {
        net->biases[layer][j] += learning_rate * deltas[layer + 1][j];
    }
}
```

---

## Training Process

### High-Level Algorithm (nn.c:318-331)

```c
void train_nn_multiclass(double **inputs, double **expected, int num_samples,
                         Net *net, int rounds, double learning_rate) {
    for (int round = 0; round < rounds; round++) {
        int *order = init_order_array(num_samples);  // Shuffle order

        for (int i = 0; i < num_samples; i++) {
            forward_pass(inputs[order[i]], net);              // Predict
            backward_pass(inputs[order[i]], expected[order[i]], // Learn
                         net, learning_rate);
        }

        free(order);
    }
}
```

### Training Loop (digits.c:73-76)

```c
train_nn_multiclass(inputs, expected, num_samples, net,
                   config->epochs, config->learning_rate);
// Trains for 50,000 epochs with learning rate 0.5
```

### Epoch Breakdown

**One Epoch = One pass through all training samples**

```
Epoch 1:
    Sample 3 → forward_pass → backward_pass → update weights
    Sample 7 → forward_pass → backward_pass → update weights
    Sample 1 → forward_pass → backward_pass → update weights
    ... (all 10 samples in random order)

Epoch 2:
    Sample 5 → forward_pass → backward_pass → update weights
    Sample 2 → forward_pass → backward_pass → update weights
    ...
```

**Why Shuffle?** Prevents the network from learning the order instead of the patterns.

### Convergence

```
MSE over training:
1.0 ┤
    │ ╲
0.5 ┤  ╲___
    │      ╲____
0.1 ┤           ╲_____
    │                 ╲________
0.0 ┤________________________╲________________
    0     10k    20k    30k    40k    50k epochs
```

Final MSE ≈ 0.000008 (near perfect)

---

## Model Serialization

### Binary File Format

```
Offset   Size    Field
──────────────────────────────────────
0x00     4       Magic number: "ENEN" (0x4E454E45)
0x04     4       Version: 1
0x08     4       Number of layers (e.g., 4)
0x0C     4×N     Layer sizes [64, 128, 64, 10]
         8×W     All weights (17,024 doubles)
         8×B     All biases (202 doubles)
```

### Save Network (nn.c:176-214)

```c
int save_net(const Net *net, const char *filename) {
    FILE *f = fopen(filename, "wb");

    // Write header
    uint32_t magic = 0x4E454E45;  // "ENEN"
    uint32_t version = 1;
    fwrite(&magic, sizeof(uint32_t), 1, f);
    fwrite(&version, sizeof(uint32_t), 1, f);

    // Write architecture
    uint32_t num_layers = (uint32_t)net->num_layers;
    fwrite(&num_layers, sizeof(uint32_t), 1, f);
    for (int i = 0; i < net->num_layers; i++) {
        int32_t size = (int32_t)net->layer_sizes[i];
        fwrite(&size, sizeof(int32_t), 1, f);
    }

    // Write weights (flattened)
    for (int layer = 0; layer < net->num_layers - 1; layer++) {
        int weight_count = net->layer_sizes[layer] * net->layer_sizes[layer + 1];
        fwrite(net->weights[layer], sizeof(double), weight_count, f);
    }

    // Write biases
    for (int layer = 0; layer < net->num_layers - 1; layer++) {
        fwrite(net->biases[layer], sizeof(double), net->layer_sizes[layer + 1], f);
    }

    fclose(f);
    return 0;
}
```

### Load Network (nn.c:216-316)

```c
Net* load_net(const char *filename) {
    FILE *f = fopen(filename, "rb");

    // Verify magic number
    uint32_t magic, version;
    fread(&magic, sizeof(uint32_t), 1, f);
    if (magic != 0x4E454E45) {
        fprintf(stderr, "Invalid model file format (expected ENEN magic)\n");
        fclose(f);
        return NULL;
    }

    // Read architecture and allocate network
    // ... (similar structure to save_net in reverse)

    return net;
}
```

**Usage:**
```bash
./digits                    # Train and save to digits.model
./digits --load digits.model # Load and test without training
```

---

## Multi-Threading

### OpenMP Parallelization

The network uses OpenMP to parallelize computations across CPU cores.

**Compiler Flags (Makefile:3-4):**
```makefile
CFLAGS = -g -Wall -fopenmp
LDFLAGS = -lm -fopenmp
```

### Parallel Forward Pass (nn.c:79)

```c
#pragma omp parallel for schedule(static)
for (int j = 0; j < next_size; j++) {
    // Each neuron computed independently by different threads
    double sum = net->biases[layer][j];
    for (int i = 0; i < current_size; i++) {
        sum += net->activations[layer][i] * net->weights[layer][i * next_size + j];
    }
    net->activations[layer + 1][j] = sigmoid(sum);
}
```

**Why This Works:**
- Each neuron's computation is independent
- No race conditions (each thread writes to different memory locations)
- Static scheduling for load balancing

### Thread Count (digits.c:102-104)

```c
#ifdef _OPENMP
    int num_threads = omp_get_max_threads();
    printf("Using %d OpenMP threads\n", num_threads);
#endif
```

**Performance:**
- Single-threaded: ~100% of one core
- Multi-threaded (16 cores): Speeds up by factor of 8-12× (not linear due to memory bandwidth)

---

## Code Reference Guide

### Core Functions

| Function | File | Line | Purpose |
|----------|------|------|---------|
| `create_net()` | nn.c | 9-50 | Allocate and initialize network |
| `free_net()` | nn.c | 52-68 | Free network memory |
| `forward_pass()` | nn.c | 73-86 | Compute network output |
| `backward_pass()` | nn.c | 90-143 | Compute gradients and update weights |
| `train_nn_multiclass()` | nn.c | 318-331 | Training loop |
| `test_nn_multiclass()` | nn.c | 333-369 | Test and report accuracy |
| `save_net()` | nn.c | 176-214 | Serialize model to disk |
| `load_net()` | nn.c | 216-316 | Deserialize model from disk |

### Dataset Functions

| Function | File | Line | Purpose |
|----------|------|------|---------|
| `load_dataset_multiclass()` | tools.c | 126-174 | Load CSV with one-hot encoding |
| `free_dataset_multiclass()` | tools.c | 176-182 | Free dataset memory |

### Utility Functions

| Function | File | Line | Purpose |
|----------|------|------|---------|
| `sigmoid()` | tools.c | 60-62 | Activation function |
| `sigmoid_derivative()` | tools.c | 64-66 | Gradient of sigmoid |
| `randinit()` | tools.c | 51-53 | Random weight initialization |
| `init_order_array()` | tools.c | 126-140 | Shuffle training order |

### Configuration

| Function | File | Line | Purpose |
|----------|------|------|---------|
| `load_config()` | config.c | - | Parse .conf file |
| `free_config()` | config.c | - | Free config memory |

---

## Testing and Validation

### Test Classification (nn.c:333-369)

```c
void test_nn_multiclass(double **inputs, double **expected, int num_samples, Net *net) {
    int correct = 0;

    for (int i = 0; i < num_samples; i++) {
        forward_pass(inputs[i], net);

        // Find predicted class (argmax)
        int predicted = 0;
        double max_activation = net->activations[output_layer][0];
        for (int j = 1; j < output_size; j++) {
            if (net->activations[output_layer][j] > max_activation) {
                max_activation = net->activations[output_layer][j];
                predicted = j;
            }
        }

        // Find expected class (argmax of one-hot)
        int expected_class = 0;
        for (int j = 1; j < output_size; j++) {
            if (expected[i][j] > expected[i][expected_class]) {
                expected_class = j;
            }
        }

        if (predicted == expected_class) correct++;
        printf("Sample %d: predicted=%d, expected=%d [%s]\n",
               i, predicted, expected_class,
               predicted == expected_class ? "OK" : "FAIL");
    }

    printf("Accuracy: %d/%d (%.1f%%)\n", correct, num_samples,
           100.0 * correct / num_samples);
}
```

### Mean Squared Error (nn.c:371-384)

```c
double test_nn_and_get_mse_multiclass(double **inputs, double **expected,
                                      int num_samples, Net *net) {
    double mse = 0.0;
    for (int i = 0; i < num_samples; i++) {
        forward_pass(inputs[i], net);

        // Sum squared error across all 10 output neurons
        for (int j = 0; j < output_size; j++) {
            double error = expected[i][j] - net->activations[output_layer][j];
            mse += error * error;
        }
    }
    return mse / num_samples;
}
```

**Interpretation:**
- MSE < 0.001: Excellent (near perfect classification)
- MSE < 0.01: Good
- MSE > 0.1: Poor (needs more training or architecture adjustment)

---

## Complete Example Workflow

### 1. Generate Dataset
```bash
./generate_digits
# Creates digits_dataset.csv with 8×8 patterns for 0-9
```

### 2. Train Network
```bash
./digits
# Output:
# Network Architecture: 64-128-64-10
# Using 16 OpenMP threads
# Training network for 50000 epochs...
# Training complete!
# Final MSE: 0.000008
#
# Sample 0: predicted=0, expected=0 [OK]
# Sample 1: predicted=1, expected=1 [OK]
# ...
# Accuracy: 10/10 (100.0%)
#
# Saving model to digits.model...
# Model saved successfully
```

### 3. Load and Test
```bash
./digits --load digits.model
# Loads pre-trained model and tests without training
# Produces identical results instantly
```

### 4. Analyze with Other Tools
```bash
./accuracy --load digits.model  # Test accuracy
./gym --load digits.model       # Try fine-tuning with different hyperparameters
```

---

## Performance Characteristics

### Training Time
- **Single-threaded:** ~30-40 seconds (50k epochs, 10 samples)
- **Multi-threaded (16 cores):** ~5-8 seconds
- **Speedup:** ~5-7× (limited by memory bandwidth, not CPU)

### Memory Usage
- **Network:** ~135 KB (17,226 parameters × 8 bytes)
- **Dataset:** ~1 KB (10 samples × 74 doubles)
- **Training buffers:** ~10 KB
- **Total:** < 1 MB

### Accuracy
- **Training set:** 100% (10/10 samples)
- **MSE:** ~0.000008
- **Convergence:** ~30,000-40,000 epochs

---

## Key Design Decisions

### Why Sigmoid?
- Smooth, differentiable everywhere
- Output in (0,1) interpretable as probability
- Simple derivative: σ'(x) = σ(x)(1-σ(x))

**Alternative:** ReLU (faster, but different training dynamics)

### Why Stochastic Gradient Descent?
- Updates after each sample (online learning)
- Faster convergence for small datasets
- Natural regularization from noise

**Alternative:** Mini-batch (more stable gradients)

### Why Row-Major Weight Storage?
```c
weight[i * next_size + j]  // Cache-friendly access pattern
```
Accessing all weights for neuron j is contiguous in memory.

### Why One-Hot Encoding?
- Treats classes as independent (no ordering assumption)
- Works naturally with cross-entropy loss (implicitly used via MSE)
- Standard approach for multi-class classification

---

## Extending the System

### Add More Layers
Edit `digits.conf`:
```
hidden_layers=256,128,64,32  # 4 hidden layers instead of 2
```

### Change Activation Function
Replace `sigmoid()` in `forward_pass()` with:
- ReLU: `max(0, x)`
- Tanh: `tanh(x)`
- Leaky ReLU: `max(0.01*x, x)`

### Add Regularization
In `backward_pass()`, add L2 penalty:
```c
net->weights[layer][idx] += learning_rate * (delta - lambda * weight);
//                                            ^^^^^^^^^^^^^^^^^
//                                            L2 regularization
```

### Larger Dataset
The system scales to larger datasets automatically. Just provide more samples in the CSV file.

---

## Common Issues and Debugging

### Network Not Learning (MSE stays high)
- **Check learning rate:** Too high → oscillation, too low → slow convergence
- **Check initialization:** Verify weights are random, not all zeros
- **Check data:** Ensure one-hot encoding is correct

### Accuracy < 100% on Training Set
- **Increase epochs:** Try 100,000 instead of 50,000
- **Adjust architecture:** Add more neurons or layers
- **Check data quality:** Verify patterns are distinct

### Segmentation Fault
- **Check array bounds:** Ensure layer_sizes match dataset
- **Verify malloc:** Check all allocations succeeded
- **Use valgrind:** `valgrind ./digits` to detect memory errors

### Model File Corrupt
- **Check magic number:**
```bash
hexdump -C digits.model | head -1
# Should show: 45 4e 45 4e (ENEN)
```

---

## Mathematical Foundation

### Forward Pass (Matrix Form)

```
a^(L+1) = σ(W^(L) · a^(L) + b^(L))

Where:
  a^(L) = activation vector at layer L
  W^(L) = weight matrix from layer L to L+1
  b^(L) = bias vector for layer L+1
  σ = sigmoid activation function
```

### Backward Pass (Gradient Descent)

```
∂E/∂w_ij = δ_j · a_i

Where:
  E = error (loss function)
  w_ij = weight from neuron i to neuron j
  δ_j = error gradient at neuron j
  a_i = activation of neuron i

Weight update:
  w_ij ← w_ij + η · δ_j · a_i

  η = learning rate
```

### Chain Rule (Backpropagation Core)

```
Output layer:
  δ_j = (expected_j - actual_j) · σ'(z_j)

Hidden layers:
  δ_i = [Σ_j (δ_j · w_ij)] · σ'(z_i)
```

---

## References

### Source Files
- **nn.c** - Neural network implementation (386 lines)
- **nn.h** - Network API (30 lines)
- **digits.c** - Digit classifier program (130 lines)
- **tools.c** - Utilities and dataset loading (184 lines)
- **generate_digits.c** - Dataset generator (142 lines)

### External Resources
- **Backpropagation:** Rumelhart, Hinton, Williams (1986)
- **Gradient Descent:** Cauchy (1847)
- **OpenMP Specification:** openmp.org

### Build System
```bash
make digits          # Build digit classifier
make all            # Build all programs
make clean          # Remove binaries
```

---

*This document describes the implementation as of 2026-02-16.*

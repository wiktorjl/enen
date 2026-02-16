# Building a Neural Network from Scratch: A Complete Walkthrough

**Learn how neural networks actually work by building a handwritten digit classifier in pure C**

---

## Introduction: Understanding by Building

You've probably heard that neural networks are "inspired by the brain" or that they're "black boxes that magically learn patterns." But how do they really work? Not at a hand-wavy level—at the level of actual code, actual math, actual data structures.

This guide walks through a complete neural network implementation in C that recognizes handwritten 8×8 pixel digits (0-9). No TensorFlow, no PyTorch, no magic. Just arrays, loops, and the math that makes learning possible.

By the end, you'll understand:
- What forward propagation actually does (and why)
- How backpropagation computes gradients (with real examples)
- Why networks need multiple layers
- How 17,000 parameters learn to classify digits

**Prerequisites:** Basic C programming, high school calculus (derivatives), and curiosity about how things work.

---

## Part 1: The Problem

### What Are We Building?

Our network will classify 8×8 pixel representations of handwritten digits. Each digit is a 64-value array of 0s and 1s representing black and white pixels:

```
Digit "3":                    Input Array:
┌────────┐                    [0,0,1,1,1,1,0,0,
│  ████  │                     0,0,0,0,0,1,0,0,
│     █  │                     0,0,1,1,1,1,0,0,
│  ████  │      ──────>        0,0,0,0,0,1,0,0,
│     █  │                     0,0,0,0,0,1,0,0,
│  ████  │                     0,0,1,1,1,1,0,0,
└────────┘                     0,0,0,0,0,0,0,0,
                                0,0,0,0,0,0,0,0]
```

The network must output which digit it sees (0-9). This is a **classification problem**: given an input, assign it to one of several categories.

### Why This Problem?

1. **Non-trivial**: Can't be solved with simple rules or linear models
2. **Visual**: Easy to understand what the network is learning
3. **Small enough**: We can trace every calculation by hand
4. **Real patterns**: Demonstrates genuine learning, not memorization

---

## Part 2: Architecture - The Network's Structure

### The Big Picture

Our network has **four layers**:

```
Input Layer     Hidden Layer 1   Hidden Layer 2   Output Layer
(64 neurons)    (128 neurons)    (64 neurons)     (10 neurons)
     │               │                │                │
  [pixels]      [patterns]       [features]        [digits]
     ↓               ↓                ↓                ↓
  ┌───┐          ┌───┐            ┌───┐            ┌───┐
  │ · │──────────│ · │────────────│ · │────────────│ 0 │  0.01
  │ · │──────────│ · │────────────│ · │────────────│ 1 │  0.02
  │ · │──────────│ · │────────────│ · │────────────│ 2 │  0.03
  │...│    ...   │...│    ...     │...│    ...     │ 3 │  0.92  ← Predicted "3"
  │ · │──────────│ · │────────────│ · │────────────│ 4 │  0.01
  │ · │──────────│ · │────────────│ · │────────────│...│  ...
  └───┘          └───┘            └───┘            └───┘
```

### What Each Layer Does

**Input Layer (64 neurons):**
Simply holds the pixel values. No computation here—just the data.

**Hidden Layer 1 (128 neurons):**
Detects **low-level patterns**—edges, corners, small curves. More neurons than inputs because we're expanding into a richer representation space.

**Hidden Layer 2 (64 neurons):**
Combines low-level patterns into **higher-level features**—loops, vertical lines, curves that distinguish "3" from "8".

**Output Layer (10 neurons):**
Makes the final **classification decision**. Each neuron represents confidence for one digit (0-9).

### The Numbers

```
Total Parameters:
- Weights: (64×128) + (128×64) + (64×10) = 17,024
- Biases:  128 + 64 + 10 = 202
────────────────────────────────────────────────
Total:     17,226 trainable parameters
```

That's 17,226 numbers that the network must learn to set correctly!

---

## Part 3: Data Structures - How It's Represented

### The Net Structure

```c
typedef struct NetStruct {
    int num_layers;          // 4 (input + 2 hidden + output)
    int *layer_sizes;        // [64, 128, 64, 10]

    double **weights;        // 3 weight matrices
    double **biases;         // 3 bias vectors
    double **activations;    // 4 activation vectors
} Net;
```

### Memory Layout

**Weights** - Stored as flattened 2D matrices:
```
weights[0]: 64→128 connections = 8,192 doubles (64 KB)
weights[1]: 128→64 connections = 8,192 doubles (64 KB)
weights[2]: 64→10 connections  = 640 doubles   (5 KB)
```

Why flattened? Cache efficiency. Accessing `weight[i][j]` becomes:
```c
weight = weights[layer][i * next_layer_size + j]
```

**Activations** - The "signal" flowing through the network:
```
activations[0]: Input layer      [64 doubles] - pixel values
activations[1]: Hidden layer 1   [128 doubles] - after computation
activations[2]: Hidden layer 2   [64 doubles] - after computation
activations[3]: Output layer     [10 doubles] - final predictions
```

These are **cached during forward propagation** and reused during backpropagation.

---

## Part 4: Forward Propagation - Making a Prediction

### The Core Algorithm

Forward propagation answers: "Given this input, what does the network predict?"

**For each layer transition:**
1. Compute weighted sum of inputs
2. Add bias term
3. Apply activation function
4. Use result as input to next layer

### The Math

For neuron `j` in layer `L+1`:

```
Step 1: Weighted Sum
z_j = Σ(activation_i × weight_ij) + bias_j
      ↑                              ↑
  Sum over all neurons          One per neuron
  in previous layer

Step 2: Activation
activation_j = sigmoid(z_j) = 1 / (1 + e^(-z_j))
```

### The Code

Here's the actual implementation from `nn.c:75-90`:

```c
void forward_pass(const double *inputs, Net *net) {
    // Step 1: Copy input pixels to first layer
    memcpy(net->activations[0], inputs,
           sizeof(double) * net->layer_sizes[0]);

    // Step 2: Propagate through each layer
    for (int layer = 0; layer < net->num_layers - 1; layer++) {
        int current_size = net->layer_sizes[layer];
        int next_size = net->layer_sizes[layer + 1];

        // Compute each neuron in next layer
        for (int j = 0; j < next_size; j++) {
            // Start with bias
            double sum = net->biases[layer][j];

            // Add weighted contributions from previous layer
            for (int i = 0; i < current_size; i++) {
                sum += net->activations[layer][i] *
                       net->weights[layer][i * next_size + j];
            }

            // Apply activation function
            net->activations[layer + 1][j] = sigmoid(sum);
        }
    }
}
```

### Worked Example: Digit "3"

Let's trace one neuron through Hidden Layer 1:

```
Input: [0,0,1,1,1,1,0,0, 0,0,0,0,0,1,0,0, ...] (64 pixels)

Hidden Neuron #42:
  sum = bias[42] = -0.234
  sum += 0.0 × weight[0→42] = -0.234
  sum += 0.0 × weight[1→42] = -0.234
  sum += 1.0 × weight[2→42] = -0.034  ← pixel #2 is ON
  sum += 1.0 × weight[3→42] =  0.266  ← pixel #3 is ON
  ... (60 more pixels)
  sum = 2.145

  activation[42] = sigmoid(2.145) = 0.895
                                     ↑
                            Neuron strongly activated!
```

This neuron "fires" because the pattern of pixels matches weights it learned during training.

### The Sigmoid Function

Why sigmoid? It **squashes** any input to the range (0, 1):

```c
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}
```

**Properties:**
- `sigmoid(-10) ≈ 0.000` (neuron OFF)
- `sigmoid(0) = 0.5` (neuron uncertain)
- `sigmoid(10) ≈ 1.000` (neuron ON)
- Smooth and differentiable (crucial for backpropagation!)

```
Sigmoid Graph:
1.0 ┤         ╭─────
    │       ╭─╯
0.5 ┤     ╭─╯
    │   ╭─╯
0.0 ┤───╯
    └────┴────┴────┴───
      -10  -5   0   5  10
```

### One-Hot Encoding for Output

The network outputs 10 numbers (one per digit). We interpret this as **confidence scores**:

```
Network Output:              Interpretation:
[0.01, 0.02, 0.03, 0.92,    "I'm 92% confident this is a '3',
 0.01, 0.01, 0.00, 0.00,     2% confident it's a '1', etc."
 0.00, 0.00]

Prediction: argmax = 3  ← The digit with highest confidence
```

During training, we represent the correct answer as a **one-hot vector**:

```
Label: 3

One-Hot Encoding:
[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
         ↑
      Only this is 1.0, rest are 0.0
```

This gives the network a clear target: "Make neuron #3 output 1.0, all others output 0.0."

---

## Part 5: The Learning Challenge

We just saw how a network makes predictions. But how do we get those 17,226 weights and biases to the right values?

### The Problem

Initially, weights are **random**:

```c
double random_uniform_init() {
    return 2.0 * ((double)rand() / RAND_MAX) - 1.0;  // Range: [-1, 1]
}
```

With random weights, the network outputs garbage:

```
Input: digit "3"
Expected: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
Actual:   [0.42, 0.38, 0.55, 0.21, 0.47, 0.39, 0.44, 0.52, 0.31, 0.48]
                              ↑
                         Completely wrong!
```

We need to **adjust each weight** to reduce this error. But which way? By how much?

### Enter: Backpropagation

**Backpropagation** solves this problem by computing the **gradient** of the error with respect to each weight:

```
∂Error/∂weight_ij = "If I increase this weight by a tiny amount,
                     how much does the error increase or decrease?"
```

If the gradient is:
- **Positive**: Increasing the weight increases error → decrease the weight
- **Negative**: Increasing the weight decreases error → increase the weight

---

## Part 6: Backpropagation - How Learning Works

Backpropagation has three phases: compute output error, propagate error backward, then update weights.

### Phase 1: Output Layer Error

**Goal:** Measure how wrong each output neuron is.

```c
// For each output neuron
for (int i = 0; i < output_size; i++) {
    double error = expected[i] - actual[i];
    deltas[output][i] = error * sigmoid_derivative(actual[i]);
}
```

**Example (digit "3"):**

```
Expected: [0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0]
Actual:   [0.01, 0.02, 0.03, 0.85, 0.01, 0.01, 0.00, 0.00, 0.00, 0.00]

Errors:   [-0.01, -0.02, -0.03, +0.15, -0.01, -0.01, 0, 0, 0, 0]
                                  ↑
                            Positive! Need to increase this.
```

The **delta** (error gradient) combines:
- Raw error: `expected - actual`
- Sigmoid derivative: `activation × (1 - activation)`

Why sigmoid derivative? The chain rule of calculus. The derivative tells us how sensitive the neuron's output is to changes in its input.

```c
double sigmoid_derivative(double sigmoid_output) {
    return sigmoid_output * (1.0 - sigmoid_output);
}
```

### Phase 2: Hidden Layer Error Propagation

**Goal:** Figure out how much each hidden neuron contributed to the output error.

```c
// For each hidden layer (backward)
for (int layer = num_layers - 2; layer >= 0; layer--) {
    for (int i = 0; i < layer_size; i++) {
        double error = 0.0;

        // Accumulate error from next layer
        for (int j = 0; j < next_layer_size; j++) {
            error += deltas[layer+1][j] * weights[layer][i * next_size + j];
        }

        deltas[layer][i] = error * sigmoid_derivative(activations[layer][i]);
    }
}
```

**Intuition:** A hidden neuron's error is the **weighted sum** of errors from neurons it connects to.

```
Hidden Neuron #42 connects to output neurons:

  Error from Output Neuron #0: delta[0] × weight[42→0] =  0.01 × 0.234 =  0.0023
  Error from Output Neuron #1: delta[1] × weight[42→1] = -0.02 × 0.123 = -0.0025
  Error from Output Neuron #2: delta[2] × weight[42→2] =  0.03 × 0.567 =  0.0170
  Error from Output Neuron #3: delta[3] × weight[42→3] =  0.15 × 0.892 =  0.1338
  ... (neurons 4-9)
  ────────────────────────────────────────────────────────────────────────────
  Total error for neuron #42: 0.1506
```

This is the **chain rule** in action—propagating derivatives backward through the network.

### Phase 3: Weight Updates

**Goal:** Adjust each weight to reduce error.

```c
// For each weight
for (int layer = 0; layer < num_layers - 1; layer++) {
    for (int i = 0; i < current_size; i++) {
        for (int j = 0; j < next_size; j++) {
            weights[layer][i * next_size + j] +=
                learning_rate * deltas[layer+1][j] * activations[layer][i];
        }
    }

    // For each bias
    for (int j = 0; j < next_size; j++) {
        biases[layer][j] += learning_rate * deltas[layer+1][j];
    }
}
```

**The Update Rule:**

```
weight_new = weight_old + learning_rate × delta × activation

Where:
  learning_rate = step size (e.g., 0.5)
  delta = error gradient for next neuron
  activation = output of current neuron
```

**Why this works:**

1. **Delta tells direction**: Positive delta means increase weight, negative means decrease
2. **Activation tells relevance**: If current neuron is OFF (activation ≈ 0), this weight doesn't matter much
3. **Learning rate controls magnitude**: Too large = unstable, too small = slow

**Example:**

```
Weight from Hidden Neuron #42 to Output Neuron #3:

  Current weight: 0.892
  Delta (output neuron #3): 0.15 (needs to increase)
  Activation (neuron #42): 0.895 (strongly ON)
  Learning rate: 0.5

  Update = 0.5 × 0.15 × 0.895 = 0.067

  New weight = 0.892 + 0.067 = 0.959
                              ↑
                      Increased! Next time, neuron #3 will
                      activate MORE for this pattern.
```

---

## Part 7: The Complete Training Loop

Now we combine everything into the training algorithm.

### Stochastic Gradient Descent

```c
void train_nn_multiclass(double **inputs, double **expected,
                         int num_samples, Net *net,
                         int epochs, double learning_rate) {
    for (int epoch = 0; epoch < epochs; epoch++) {
        // Shuffle training examples
        int *order = init_order_array(num_samples);

        // Train on each example
        for (int i = 0; i < num_samples; i++) {
            forward_pass(inputs[order[i]], net);
            backward_pass(inputs[order[i]], expected[order[i]],
                         net, learning_rate);
        }

        free(order);
    }
}
```

### One Epoch = One Full Pass

With 10 training examples (digits 0-9), one epoch looks like:

```
Epoch 1:
  Sample 7 → forward → backward → update weights
  Sample 2 → forward → backward → update weights
  Sample 9 → forward → backward → update weights
  Sample 0 → forward → backward → update weights
  ... (6 more, random order)

Epoch 2:
  Sample 3 → forward → backward → update weights
  Sample 1 → forward → backward → update weights
  ...
```

### Why Shuffle?

Without shuffling, the network might learn the **order** instead of the patterns:

```
Fixed order: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
Network might learn: "After seeing '2', expect '3'"

Random order: [7, 2, 9, 0, 4, ...]
Network must learn: "What digit is THIS pattern?"
```

### Convergence

Over many epochs, the error gradually decreases:

```
Mean Squared Error over Training:

1.0 ┤ ●
0.9 ┤ ●
0.8 ┤  ●
    │   ●
0.5 ┤    ●●
    │      ●●
0.2 ┤        ●●●
0.1 ┤           ●●●●●
    │                ●●●●●●●●●●●●
0.0 ┤____________________________________●●●●●●●●●●
    0    5k   10k   15k   20k   30k   40k   50k  epochs
```

After 50,000 epochs:
- **MSE**: 0.000008 (near perfect)
- **Accuracy**: 100% on all 10 training samples

---

## Part 8: Testing and Evaluation

### Making Predictions

After training, we test the network:

```c
void test_nn_multiclass(double **inputs, double **expected,
                        int num_samples, Net *net) {
    int correct = 0;

    for (int i = 0; i < num_samples; i++) {
        forward_pass(inputs[i], net);

        // Find predicted class (highest activation)
        int predicted = argmax(activations[output_layer], 10);
        int expected_class = argmax(expected[i], 10);

        if (predicted == expected_class) correct++;

        printf("Sample %d: predicted=%d, expected=%d [%s]\n",
               i, predicted, expected_class,
               predicted == expected_class ? "OK" : "FAIL");
    }

    printf("\nAccuracy: %d/%d (%.1f%%)\n",
           correct, num_samples, 100.0 * correct / num_samples);
}
```

### Sample Output

```
=== Classification Results ===
Sample 0: predicted=0, expected=0 [OK]
Sample 1: predicted=1, expected=1 [OK]
Sample 2: predicted=2, expected=2 [OK]
Sample 3: predicted=3, expected=3 [OK]
Sample 4: predicted=4, expected=4 [OK]
Sample 5: predicted=5, expected=5 [OK]
Sample 6: predicted=6, expected=6 [OK]
Sample 7: predicted=7, expected=7 [OK]
Sample 8: predicted=8, expected=8 [OK]
Sample 9: predicted=9, expected=9 [OK]

Accuracy: 10/10 (100.0%)
```

### Mean Squared Error

```c
double test_nn_and_get_mse_multiclass(double **inputs, double **expected,
                                      int num_samples, Net *net) {
    double mse = 0.0;

    for (int i = 0; i < num_samples; i++) {
        forward_pass(inputs[i], net);

        // Sum squared error across all 10 outputs
        for (int j = 0; j < 10; j++) {
            double error = expected[i][j] - activations[output][j];
            mse += error * error;
        }
    }

    return mse / num_samples;
}
```

MSE measures average squared difference between expected and actual outputs. Lower is better:
- **MSE < 0.001**: Excellent
- **MSE < 0.01**: Good
- **MSE > 0.1**: Poor (needs more training)

---

## Part 9: Model Persistence

### Saving the Network

After training for 50,000 epochs, we can save the weights to disk:

```c
int save_net(const Net *net, const char *filename) {
    FILE *f = fopen(filename, "wb");

    // Write magic number for verification
    uint32_t magic = 0x4E454E45;  // "ENEN"
    uint32_t version = 1;
    fwrite(&magic, sizeof(uint32_t), 1, f);
    fwrite(&version, sizeof(uint32_t), 1, f);

    // Write architecture
    fwrite(&net->num_layers, sizeof(uint32_t), 1, f);
    for (int i = 0; i < net->num_layers; i++) {
        fwrite(&net->layer_sizes[i], sizeof(int32_t), 1, f);
    }

    // Write all weights (17,024 doubles)
    for (int layer = 0; layer < net->num_layers - 1; layer++) {
        int count = layer_sizes[layer] * layer_sizes[layer+1];
        fwrite(net->weights[layer], sizeof(double), count, f);
    }

    // Write all biases (202 doubles)
    for (int layer = 0; layer < net->num_layers - 1; layer++) {
        fwrite(net->biases[layer], sizeof(double),
               layer_sizes[layer+1], f);
    }

    fclose(f);
    return 0;
}
```

**Binary File Format:**

```
Offset   Size        Field
──────────────────────────────────────────
0x00     4 bytes     Magic: "ENEN" (0x4E454E45)
0x04     4 bytes     Version: 1
0x08     4 bytes     Num layers: 4
0x0C     16 bytes    Layer sizes: [64, 128, 64, 10]
0x1C     136,192 B   Weights (17,024 × 8 bytes)
...      1,616 B     Biases (202 × 8 bytes)
──────────────────────────────────────────
Total:   ~135 KB
```

### Loading and Testing

```bash
# Train and save
$ ./build/digits
Training network for 50000 epochs...
Training complete! Final MSE: 0.000008
Saving model to models/digits.model...

# Load and test instantly
$ ./build/digits --load models/digits.model
Loading model from models/digits.model...
Model loaded successfully
Network Architecture: 64-128-64-10

Sample 0: predicted=0, expected=0 [OK]
Sample 1: predicted=1, expected=1 [OK]
...
Accuracy: 10/10 (100.0%)
```

Loading takes milliseconds vs. 5-8 seconds of training!

---

## Part 10: Performance and Parallelization

### OpenMP Multi-Threading

The network uses OpenMP to parallelize neuron computations:

```c
#pragma omp parallel for schedule(static)
for (int j = 0; j < next_size; j++) {
    // Each thread computes a different neuron
    double sum = biases[layer][j];
    for (int i = 0; i < current_size; i++) {
        sum += activations[layer][i] * weights[layer][i * next_size + j];
    }
    activations[layer + 1][j] = sigmoid(sum);
}
```

**Why This Works:**
- Each neuron's computation is **independent**
- No race conditions (each thread writes to different memory)
- Static scheduling for predictable load balancing

### Performance Numbers

**Training Time (50,000 epochs, 10 samples):**
```
Single-threaded:     ~35 seconds
16 cores (OpenMP):   ~6 seconds
Speedup:             5.8×
```

Not a full 16× speedup because:
- Memory bandwidth becomes the bottleneck
- Thread synchronization overhead
- Cache contention

**Memory Usage:**
```
Network structure:    ~135 KB
Training buffers:     ~10 KB
Dataset:              ~1 KB
────────────────────────────
Total:                < 1 MB
```

Tiny by modern standards!

---

## Part 11: What You've Learned

Congratulations! You now understand neural networks at a deep level:

### Core Concepts

**1. Forward Propagation**
- How networks make predictions layer by layer
- The role of weights, biases, and activation functions
- Why we use sigmoid (smooth, differentiable, bounded)

**2. Backpropagation**
- How to compute gradients efficiently using the chain rule
- Error propagation from output to input
- The update rule: `weight += learning_rate × gradient × input`

**3. Training Process**
- Stochastic gradient descent with shuffling
- Why we need multiple epochs
- How error decreases over time (convergence)

**4. Network Architecture**
- Why hidden layers matter (non-linear transformations)
- How layer sizes affect capacity
- The tradeoff between expressiveness and trainability

### Implementation Details

You've seen actual production code showing:
- Dynamic memory allocation for arbitrary network sizes
- Efficient weight indexing (row-major matrices)
- Model serialization for persistence
- Multi-threaded computation with OpenMP

### The Math

You've worked through:
- The sigmoid function and its derivative
- The chain rule for backpropagation
- Gradient descent weight updates
- Mean squared error as a loss function

---

## Part 12: Going Further

### Extending This Implementation

**1. Different Activation Functions**

Try ReLU instead of sigmoid:
```c
double relu(double x) {
    return x > 0 ? x : 0;
}

double relu_derivative(double x) {
    return x > 0 ? 1 : 0;
}
```

ReLU trains faster but can "die" (outputs become stuck at zero).

**2. Better Weight Initialization**

The code includes Xavier initialization:
```c
double xavier_init(int fan_in, int fan_out) {
    double limit = sqrt(6.0 / (fan_in + fan_out));
    return limit * (2.0 * rand()/RAND_MAX - 1.0);
}
```

This prevents vanishing/exploding gradients in deep networks.

**3. Regularization**

Add L2 penalty to prevent overfitting:
```c
// In weight update
weight += learning_rate * (delta * activation - lambda * weight);
                                                 ↑
                                           Pulls weights toward zero
```

**4. Learning Rate Scheduling**

Reduce learning rate over time:
```c
double lr = initial_lr * (1.0 / (1.0 + decay_rate * epoch));
```

Helps fine-tune in later epochs.

### Real-World Extensions

To recognize real handwriting (28×28 MNIST):
1. Increase input size to 784 neurons
2. Add more/larger hidden layers (e.g., 512-256-128)
3. Use ReLU activation
4. Implement mini-batch training
5. Add momentum or Adam optimizer
6. Use cross-entropy loss instead of MSE

With these changes, you'd achieve ~98% accuracy on MNIST!

---

## Part 13: Common Questions

**Q: Why not just use TensorFlow?**

A: You should! But understanding the underlying mechanics makes you better at:
- Debugging when things go wrong
- Choosing appropriate architectures
- Understanding research papers
- Building custom solutions when libraries don't fit

**Q: Is this how modern neural networks work?**

A: Yes! The fundamentals are identical:
- Forward prop → compute output
- Backward prop → compute gradients
- Gradient descent → update weights

Modern frameworks add optimizations (GPU acceleration, automatic differentiation, better optimizers), but the core algorithm is the same.

**Q: Why sigmoid instead of ReLU?**

A: Pedagogical simplicity. Sigmoid's derivative is elegant (`σ'(x) = σ(x)(1-σ(x))`), making backpropagation easier to understand. ReLU is faster and often better, but less intuitive for learning.

**Q: How do you handle overfitting with just 10 samples?**

A: We don't! This implementation memorizes the training set perfectly. For real ML:
- Split data into train/validation/test sets
- Use regularization (dropout, L2)
- Early stopping when validation error increases
- Data augmentation

**Q: What about convolutional layers for images?**

A: Fully-connected layers (what we built) work for small images. For larger images, convolutional layers are more efficient—they learn translation-invariant features and have fewer parameters. But the training algorithm (backprop + SGD) is identical!

---

## Part 14: The Code Reference

All source files with key locations:

### Core Neural Network (`nn.c`)

| Function | Lines | Purpose |
|----------|-------|---------|
| `create_net()` | 14-54 | Allocate network and initialize weights |
| `free_net()` | 57-73 | Clean up memory |
| `forward_pass()` | 75-91 | Compute predictions |
| `backward_pass()` | 93-143 | Compute gradients and update weights |
| `train_nn_multiclass()` | 318-331 | Training loop with shuffling |
| `test_nn_multiclass()` | 333-369 | Evaluate accuracy |
| `save_net()` | 185-223 | Serialize to disk |
| `load_net()` | 225-316 | Deserialize from disk |

### Utilities (`tools.c`)

| Function | Lines | Purpose |
|----------|-------|---------|
| `sigmoid()` | 60-62 | Activation function |
| `sigmoid_derivative()` | 64-66 | Gradient of sigmoid |
| `load_dataset_multiclass()` | 142-192 | Load CSV with one-hot encoding |
| `init_order_array()` | 126-140 | Shuffle training order |

### Main Program (`digits.c`)

| Section | Lines | Purpose |
|---------|-------|---------|
| Argument parsing | 30-56 | Handle `--load` and `--help` |
| Training mode | 80-131 | Train new network |
| Testing mode | 61-78 | Load and test existing model |

---

## Conclusion: You Built a Neural Network

You've just worked through every line of code and every mathematical operation in a working neural network. You understand:

- **The data structures** that represent neurons, weights, and activations
- **The forward pass** that makes predictions
- **The backward pass** that computes gradients
- **The training loop** that adjusts 17,226 parameters to learn patterns
- **The engineering** that makes it fast and persistent

This is the foundation of deep learning. CNNs, RNNs, Transformers—they all use these same principles, just with different architectures and activation functions.

Most importantly, you've seen that neural networks aren't magic. They're **structured computation** guided by **calculus** to **minimize error**. Elegant, powerful, and completely understandable.

Now go build something amazing.

---

## Appendix: Building and Running

```bash
# Build all tools
$ make

# Generate training data
$ ./build/generate_digits
Created digits_dataset.csv

# Train network
$ ./build/digits
Network Architecture: 64-128-64-10
Using 16 OpenMP threads
Training network for 50000 epochs...
Training complete!
Final MSE: 0.000008
Accuracy: 10/10 (100.0%)
Saving model to models/digits.model...
Model saved successfully

# Test saved model
$ ./build/digits --load models/digits.model
Loading model from models/digits.model...
Model loaded successfully

Sample 0: predicted=0, expected=0 [OK]
Sample 1: predicted=1, expected=1 [OK]
...
Accuracy: 10/10 (100.0%)

# Statistical analysis
$ ./build/accuracy 100
Running 100 training sessions...
Average Accuracy: 98.5%
Standard Deviation: 7.2%

# Hyperparameter tuning
$ ./build/gym
Testing learning rates and epochs...
| Learning Rate | Rounds | MSE      |
|---------------|--------|----------|
| 0.50          | 50000  | 0.000008 | ← Best
```

---

**Further Reading:**
- [Original Backpropagation Paper](https://www.nature.com/articles/323533a0) - Rumelhart, Hinton, Williams (1986)
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) - Michael Nielsen
- [Deep Learning Book](https://www.deeplearningbook.org/) - Goodfellow, Bengio, Courville

**Source Code:** Available at the repository root. MIT licensed for educational use.

---

*Last updated: February 16, 2026*

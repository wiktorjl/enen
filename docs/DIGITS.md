# How the UCI Digits Classifier Works

This document follows the code in `src/` and describes the one task implemented
by the repository: assigning an 8×8 grayscale image to one of ten digit classes.
There is no binary-classification API or alternate synthetic-data workflow.

## 1. Data and task

Each UCI optical-digits sample contains 64 pixel intensities and one class label.
The source files use integer intensities from 0 through 16. The committed model
inputs divide those intensities by 16, producing values in `[0, 1]`.

```text
x = [x0, x1, ..., x63],  where 0 <= xi <= 1
label in {0, 1, ..., 9}
```

`dataset.c` converts the integer label into a one-hot target. For a sample whose
label is 3:

```text
y = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
```

This target is explicitly multiclass: all ten output neurons participate in the
same prediction.

The repository preserves the UCI split:

| Split | Raw file | Normalized file | Samples | Responsibility |
|---|---|---|---:|---|
| Training | `optdigits.tra` | `UCI_digits_train.csv` | 3,823 | Fit weights |
| Test | `optdigits.tes` | `UCI_digits_test.csv` | 1,797 | Final evaluation only |

The `gym` program creates a further deterministic validation slice from the
training rows. It never selects hyperparameters with the test set.

### Input validation

The loader requires exactly 64 numeric feature fields followed by one integer
label. It rejects:

- missing or extra fields;
- nonnumeric and nonfinite values;
- pixels outside `[0, 1]`;
- labels outside the configured class range;
- overlong lines and empty files.

Rejecting an entire malformed file is important. Silently accepting a short row
would leave uninitialized features, and accepting an invalid label would create
a target with no true class.

## 2. Network architecture

The default configuration builds this fully connected network:

```text
64 inputs -> 128 sigmoid units -> 64 sigmoid units -> 10-way softmax
```

`Net` stores:

```c
typedef struct {
    int num_layers;
    int *layer_sizes;
    double **weights;
    double **biases;
    double **activations;
    double **deltas;
} Net;
```

A connection from input `i` to output `j` in one layer transition is stored in
a flat row-major matrix:

```c
weights[layer][i * next_size + j]
```

The parameter count is:

```text
weights = 64*128 + 128*64 + 64*10 = 17,024
biases  = 128 + 64 + 10             =    202
total                                      17,226
```

Weights are sampled from the Xavier/Glorot uniform range
`[-sqrt(6/(fan_in+fan_out)), +sqrt(6/(fan_in+fan_out))]`. Biases start at zero.
This keeps initial activation scales reasonable without embedding a preference
for any digit.

## 3. Forward propagation

For a hidden neuron `j`, the network calculates a weighted sum and applies the
sigmoid function:

```text
z_j = b_j + sum_i(a_i * w_ij)
a_j = sigmoid(z_j) = 1 / (1 + exp(-z_j))
```

The implementation uses separate positive and negative branches for sigmoid,
avoiding overflow when a weighted sum has large magnitude.

The output layer deliberately does not apply ten independent sigmoids. It first
computes ten logits and then applies softmax:

```text
p_k = exp(z_k - max(z)) / sum_j(exp(z_j - max(z)))
```

Subtracting the largest logit is numerically stable and does not change the
resulting distribution. The probabilities are positive and sum to one, so they
represent mutually exclusive digit classes. The prediction is:

```text
predicted_class = argmax_k(p_k)
```

Calling these outputs probabilities is justified by the normalization. They are
not guaranteed to be calibrated probabilities; calibration is a separate
empirical property.

## 4. Loss and backpropagation

For a one-hot target, categorical cross-entropy is the negative log-probability
assigned to the true digit:

```text
loss = -log(p_true)
```

The reported test or validation loss is the mean across samples. Accuracy and
cross-entropy answer different questions: accuracy only considers the winning
class, while cross-entropy also penalizes an uncertain or confidently wrong
distribution.

Softmax and cross-entropy simplify the derivative at the output. The usual loss
gradient with respect to a logit is `p - y`. The code stores `y - p` because its
update is written with `+=`:

```text
delta_output = y - p
weight += learning_rate * delta_output * previous_activation
```

For each hidden layer, the chain rule propagates the next layer's deltas through
the weights and multiplies by the sigmoid derivative:

```text
error_i = sum_j(delta_next_j * w_ij)
delta_i = error_i * a_i * (1 - a_i)
```

Only hidden and output deltas are needed. Computing a delta for the raw input
layer would not affect any parameter and is intentionally omitted.

## 5. Stochastic training

`train_classifier()` creates one array of sample indices, shuffles it with the
Fisher-Yates algorithm before every epoch, and processes one sample at a time:

```text
for each epoch:
    shuffle(sample_order)
    for each sample in sample_order:
        forward_pass(sample)
        backward_pass(target)
```

The default learning rate and epoch count are configuration values, not hidden
constants. They were chosen from the ranges exposed by `gym`; different model
sizes can require different values.

## 6. Evaluation

`evaluate_classifier()` makes one forward pass per sample and returns:

- total and correct prediction counts;
- mean categorical cross-entropy;
- optionally, a `num_classes × num_classes` confusion matrix.

The confusion matrix uses actual classes as rows and predicted classes as
columns. Diagonal entries are correct classifications; an off-diagonal entry at
row 3, column 8 counts a true 3 predicted as an 8.

The three executable workflows keep their responsibilities separate:

| Program | Fits on | Selects with | Reports on |
|---|---|---|---|
| `digits` | Full training split | Configuration | Test split |
| `accuracy` | Full training split per run | Configuration | Test split |
| `gym` | Four fifths of training split | Remaining fifth | Validation slice |

`gym` clones one starting network for each candidate and resets the training
shuffle seed. Consequently, candidates differ by learning rate and epoch count,
not by an accidentally easier initialization or sample order. When `--load` is
used, the saved network is that common starting point.

## 7. Model persistence

Version 1 model files contain, in order:

```text
32-bit magic value "ENEN"
32-bit version
32-bit layer count
32-bit size for each layer
all weight matrices as doubles
all bias vectors as doubles
```

The loader bounds-checks layer counts and sizes, checks every read, and rejects
truncated files or trailing data. Allocation uses zero-initialized pointer
arrays, so cleanup is safe even if loading fails partway through. Before a model
is evaluated, its input width and class count must match the dataset config.

The format is intentionally small and educational. It stores native integer and
double representations, so it is not designed as a portable interchange format
between architectures with different byte order or floating-point formats.

## 8. Configuration

`conf/digits.conf` contains:

```ini
input_size=64
hidden_layers=128,64
output_size=10
learning_rate=0.05
epochs=25
train_dataset=datasets/UCI_digits_train.csv
test_dataset=datasets/UCI_digits_test.csv
```

All seven keys are required exactly once. Blank lines and lines beginning with
`#` are ignored. Unknown keys, duplicate keys, malformed numbers, empty paths,
and missing fields produce an error rather than an implicit default.

## 9. Building, testing, and regenerating data

Build every program and run the verification suite:

```bash
make all
make check
```

The tests verify both committed datasets, every represented class, normalized
features, one-hot targets, softmax normalization, loss reduction during
training, model cloning, and a save/load round trip. They also regenerate both
normalized CSVs from the raw UCI files and compare them byte for byte.

Train and evaluate the default classifier:

```bash
./build/digits
```

Evaluate the saved model:

```bash
./build/digits --load models/digits.model
./build/accuracy --load models/digits.model
```

Run repeated accuracy measurements or validation-based tuning:

```bash
./build/accuracy 10
./build/gym
```

Regenerate normalized data explicitly:

```bash
./build/convert_optdigits \
  datasets/optdigits.tra datasets/UCI_digits_train.csv \
  datasets/optdigits.tes datasets/UCI_digits_test.csv
```

## 10. Intentional limitations

This is a compact teaching implementation. It uses dense layers and per-sample
updates; it does not implement convolutions, minibatches, regularization,
adaptive optimizers, probability calibration, or a portable model container.
Those are useful extensions, but none is necessary to understand the complete
multiclass path implemented here.

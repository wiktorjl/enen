#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nn.h"

/*
 * Road map
 * --------
 *  1. Activation and initialization helpers
 *  2. Network allocation, construction, and lifetime
 *  3. The core algorithm: forward propagation and backpropagation
 *  4. Prediction, shuffled stochastic-gradient-descent (SGD), and evaluation
 *  5. The small binary save/load format
 *
 * A Net keeps one activation and one delta array per layer. Between adjacent
 * layers, weights[layer] is a flattened matrix with this logical shape:
 *
 *     [number of neurons in layer][number of neurons in layer + 1]
 *
 * In other words, weight (source, destination) lives at
 * `source * next_size + destination`. Biases need only one value per
 * destination neuron. Keeping that convention in mind makes the three nested
 * loops in forward_pass() and backward_pass() much easier to read.
 */

#define MODEL_MAGIC UINT32_C(0x4E454E45)
#define MODEL_VERSION UINT32_C(1)
#define MAX_MODEL_LAYERS 128
#define MAX_LAYER_SIZE 1000000

/* --- Activation and initialization helpers ----------------------------- */

/* This two-branch form is sigmoid(x) without overflowing exp() at extremes. */
static double sigmoid(double value) {
    if (value >= 0.0) {
        return 1.0 / (1.0 + exp(-value));
    }
    double exponential = exp(value);
    return exponential / (1.0 + exponential);
}

static double sigmoid_derivative(double activation) {
    /* For a = sigmoid(x), the derivative is a * (1 - a). */
    return activation * (1.0 - activation);
}

static double xavier_weight(int fan_in, int fan_out) {
    /* Xavier uniform: draw from [-sqrt(6/(in+out)), +sqrt(6/(in+out))]. */
    double limit = sqrt(6.0 / (fan_in + fan_out));
    double unit = (double)rand() / (double)RAND_MAX;
    return limit * (2.0 * unit - 1.0);
}

static int valid_architecture(const int *layer_sizes, int num_layers) {
    if (!layer_sizes || num_layers < 2 || num_layers > MAX_MODEL_LAYERS) {
        return 0;
    }
    for (int layer = 0; layer < num_layers; layer++) {
        if (layer_sizes[layer] <= 0 || layer_sizes[layer] > MAX_LAYER_SIZE) {
            return 0;
        }
    }
    return layer_sizes[num_layers - 1] >= 2;
}

/* --- Network allocation, construction, and lifetime -------------------- */

/* Allocate every array, but leave learned weights for the caller to fill. */
static Net *allocate_net(const int *layer_sizes, int num_layers) {
    if (!valid_architecture(layer_sizes, num_layers)) {
        fprintf(stderr, "Invalid network architecture\n");
        return NULL;
    }

    Net *net = calloc(1, sizeof(*net));
    if (!net) {
        perror("Failed to allocate network");
        return NULL;
    }
    net->num_layers = num_layers;
    net->layer_sizes = malloc(sizeof(*net->layer_sizes) * (size_t)num_layers);
    net->weights = calloc((size_t)(num_layers - 1), sizeof(*net->weights));
    net->biases = calloc((size_t)(num_layers - 1), sizeof(*net->biases));
    net->activations = calloc((size_t)num_layers, sizeof(*net->activations));
    net->deltas = calloc((size_t)num_layers, sizeof(*net->deltas));
    if (!net->layer_sizes || !net->weights || !net->biases ||
        !net->activations || !net->deltas) {
        perror("Failed to allocate network arrays");
        free_net(net);
        return NULL;
    }
    memcpy(net->layer_sizes, layer_sizes,
           sizeof(*net->layer_sizes) * (size_t)num_layers);

    /* Activations and deltas are working memory, one vector per layer. */
    for (int layer = 0; layer < num_layers; layer++) {
        size_t layer_size = (size_t)layer_sizes[layer];
        net->activations[layer] = calloc(layer_size, sizeof(double));
        net->deltas[layer] = calloc(layer_size, sizeof(double));
        if (!net->activations[layer] || !net->deltas[layer]) {
            perror("Failed to allocate neuron arrays");
            free_net(net);
            return NULL;
        }
    }

    /*
     * Each connection owns a dense weight matrix and a destination bias
     * vector.
     */
    for (int layer = 0; layer < num_layers - 1; layer++) {
        size_t current_size = (size_t)layer_sizes[layer];
        size_t next_size = (size_t)layer_sizes[layer + 1];
        if (current_size > SIZE_MAX / next_size ||
            current_size * next_size > SIZE_MAX / sizeof(double)) {
            fprintf(stderr, "Network weight matrix is too large\n");
            free_net(net);
            return NULL;
        }
        net->weights[layer] = malloc(
            sizeof(double) * current_size * next_size);
        net->biases[layer] = calloc(next_size, sizeof(double));
        if (!net->weights[layer] || !net->biases[layer]) {
            perror("Failed to allocate network parameters");
            free_net(net);
            return NULL;
        }
    }
    return net;
}

Net *create_net(const Config *config) {
    if (!config || config->num_hidden_layers < 1 ||
        !config->hidden_layer_sizes) {
        fprintf(stderr, "Invalid network configuration\n");
        return NULL;
    }

    /* Turn input + hidden + output sizes into one uniform layer-size array. */
    int num_layers = config->num_hidden_layers + 2;
    int *layer_sizes = malloc(sizeof(*layer_sizes) * (size_t)num_layers);
    if (!layer_sizes) {
        perror("Failed to allocate architecture");
        return NULL;
    }
    layer_sizes[0] = config->input_size;
    for (int layer = 0; layer < config->num_hidden_layers; layer++) {
        layer_sizes[layer + 1] = config->hidden_layer_sizes[layer];
    }
    layer_sizes[num_layers - 1] = config->output_size;

    Net *net = allocate_net(layer_sizes, num_layers);
    free(layer_sizes);
    if (!net) {
        return NULL;
    }

    /* Biases start at zero; initialize every learned weight independently. */
    for (int layer = 0; layer < net->num_layers - 1; layer++) {
        int current_size = net->layer_sizes[layer];
        int next_size = net->layer_sizes[layer + 1];
        size_t weight_count = (size_t)current_size * (size_t)next_size;
        for (size_t index = 0; index < weight_count; index++) {
            net->weights[layer][index] =
                xavier_weight(current_size, next_size);
        }
    }
    return net;
}

Net *clone_net(const Net *source) {
    if (!source) {
        return NULL;
    }
    Net *copy = allocate_net(source->layer_sizes, source->num_layers);
    if (!copy) {
        return NULL;
    }
    /*
     * Copy learned parameters; activations and deltas remain fresh workspaces.
     */
    for (int layer = 0; layer < source->num_layers - 1; layer++) {
        int current_size = source->layer_sizes[layer];
        int next_size = source->layer_sizes[layer + 1];
        size_t weight_count = (size_t)current_size * (size_t)next_size;
        memcpy(copy->weights[layer], source->weights[layer],
               sizeof(double) * weight_count);
        memcpy(copy->biases[layer], source->biases[layer],
               sizeof(double) * (size_t)next_size);
    }
    return copy;
}

void free_net(Net *net) {
    if (!net) {
        return;
    }
    if (net->activations) {
        for (int layer = 0; layer < net->num_layers; layer++) {
            free(net->activations[layer]);
        }
    }
    if (net->deltas) {
        for (int layer = 0; layer < net->num_layers; layer++) {
            free(net->deltas[layer]);
        }
    }
    if (net->weights) {
        for (int layer = 0; layer < net->num_layers - 1; layer++) {
            free(net->weights[layer]);
        }
    }
    if (net->biases) {
        for (int layer = 0; layer < net->num_layers - 1; layer++) {
            free(net->biases[layer]);
        }
    }
    free(net->activations);
    free(net->deltas);
    free(net->weights);
    free(net->biases);
    free(net->layer_sizes);
    free(net);
}

void print_net(const Net *net) {
    if (!net) {
        return;
    }
    printf("Network architecture: ");
    for (int layer = 0; layer < net->num_layers; layer++) {
        printf(layer + 1 == net->num_layers ? "%d\n" : "%d -> ",
               net->layer_sizes[layer]);
    }
}

int net_matches_dimensions(const Net *net, int input_size, int num_classes) {
    if (!net) {
        return 0;
    }
    return net->layer_sizes[0] == input_size &&
           net->layer_sizes[net->num_layers - 1] == num_classes;
}

/* --- Core algorithm (the "meat"): forward propagation ----------------- */

/*
 * Move one example from the input layer to the output layer.
 *
 * For each destination neuron j, the affine part is
 *
 *     z_j = bias_j + sum_i(activation_i * weight_ij)
 *
 * Hidden layers store sigmoid(z_j). The output layer first stores the raw
 * scores (logits), then the final block converts all logits together into
 * class probabilities with softmax. The resulting activations sum to one.
 */
void forward_pass(const double *inputs, Net *net) {
    /* Layer zero is data, not a computed layer. */
    memcpy(net->activations[0], inputs,
           sizeof(double) * (size_t)net->layer_sizes[0]);

    int output_layer = net->num_layers - 1;
    for (int layer = 0; layer < output_layer; layer++) {
        int current_size = net->layer_sizes[layer];
        int next_size = net->layer_sizes[layer + 1];
        for (int output = 0; output < next_size; output++) {
            double sum = net->biases[layer][output];
            for (int input = 0; input < current_size; input++) {
                /* Matrix row input, column output: weight input -> output. */
                sum += net->activations[layer][input] *
                       net->weights[layer][input * next_size + output];
            }
            net->activations[layer + 1][output] =
                layer + 1 == output_layer ? sum : sigmoid(sum);
        }
    }

    /*
     * Softmax turns the output logits into one normalized class distribution.
     * Subtracting the largest logit changes no probability but keeps exp()
     * away from overflow.
     */
    int output_size = net->layer_sizes[output_layer];
    double largest_logit = net->activations[output_layer][0];
    for (int class_id = 1; class_id < output_size; class_id++) {
        if (net->activations[output_layer][class_id] > largest_logit) {
            largest_logit = net->activations[output_layer][class_id];
        }
    }
    double total = 0.0;
    for (int class_id = 0; class_id < output_size; class_id++) {
        double probability =
            exp(net->activations[output_layer][class_id] - largest_logit);
        net->activations[output_layer][class_id] = probability;
        total += probability;
    }
    for (int class_id = 0; class_id < output_size; class_id++) {
        net->activations[output_layer][class_id] /= total;
    }
}

/* --- Core algorithm (the "meat"): backpropagation --------------------- */

/*
 * Learn from the example most recently passed through forward_pass().
 *
 * This code defines delta with the update-friendly sign (target - prediction),
 * the negative of the usual loss gradient. For softmax plus cross-entropy:
 *
 *     output_delta_j = target_j - probability_j
 *
 * The chain rule moves that signal backward through each sigmoid hidden layer:
 *
 *     hidden_delta_i = a_i * (1 - a_i) * sum_j(weight_ij * next_delta_j)
 *
 * Finally SGD takes one step toward lower loss:
 *
 *     weight_ij += learning_rate * a_i * next_delta_j
 *     bias_j      += learning_rate * next_delta_j
 */
void backward_pass(const double *target, Net *net, double learning_rate) {
    int output_layer = net->num_layers - 1;
    int output_size = net->layer_sizes[output_layer];

    /* For softmax plus cross-entropy, target - probability is the output delta. */
    for (int output = 0; output < output_size; output++) {
        net->deltas[output_layer][output] =
            target[output] - net->activations[output_layer][output];
    }

    /*
     * Compute every hidden delta before changing any weight. The input layer is
     * raw data, so there is no activation function there to differentiate.
     */
    for (int layer = output_layer - 1; layer >= 1; layer--) {
        int current_size = net->layer_sizes[layer];
        int next_size = net->layer_sizes[layer + 1];
        for (int input = 0; input < current_size; input++) {
            double error = 0.0;
            for (int output = 0; output < next_size; output++) {
                error += net->deltas[layer + 1][output] *
                         net->weights[layer][input * next_size + output];
            }
            net->deltas[layer][input] =
                error * sigmoid_derivative(net->activations[layer][input]);
        }
    }

    /* Apply the parameter updates only after all backward signals are ready. */
    for (int layer = 0; layer < output_layer; layer++) {
        int current_size = net->layer_sizes[layer];
        int next_size = net->layer_sizes[layer + 1];
        for (int input = 0; input < current_size; input++) {
            for (int output = 0; output < next_size; output++) {
                net->weights[layer][input * next_size + output] +=
                    learning_rate * net->deltas[layer + 1][output] *
                    net->activations[layer][input];
            }
        }
        for (int output = 0; output < next_size; output++) {
            net->biases[layer][output] +=
                learning_rate * net->deltas[layer + 1][output];
        }
    }
}

/* --- Prediction, SGD training, and evaluation -------------------------- */

static int index_of_max(const double *values, int count) {
    int maximum = 0;
    for (int index = 1; index < count; index++) {
        if (values[index] > values[maximum]) {
            maximum = index;
        }
    }
    return maximum;
}

int predict_class(const double *inputs, Net *net) {
    forward_pass(inputs, net);
    int output_layer = net->num_layers - 1;
    return index_of_max(net->activations[output_layer],
                        net->layer_sizes[output_layer]);
}

static void shuffle_order(int *order, int count) {
    /* In-place Fisher-Yates shuffle: each epoch visits every sample once. */
    for (int index = count - 1; index > 0; index--) {
        int replacement = rand() % (index + 1);
        int temporary = order[index];
        order[index] = order[replacement];
        order[replacement] = temporary;
    }
}

int train_classifier(double **inputs, double **targets, int num_samples,
                     Net *net, int epochs, double learning_rate) {
    if (!inputs || !targets || !net || num_samples <= 0 || epochs <= 0 ||
        !isfinite(learning_rate) || learning_rate <= 0.0) {
        return -1;
    }
    int *order = malloc(sizeof(*order) * (size_t)num_samples);
    if (!order) {
        perror("Failed to allocate training order");
        return -1;
    }
    for (int sample = 0; sample < num_samples; sample++) {
        order[sample] = sample;
    }

    /*
     * This loop is the training "meat": stochastic gradient descent. Shuffle,
     * then update immediately after each single example. A mini-batch optimizer
     * would accumulate gradients before updating; this intentionally does not.
     */
    for (int epoch = 0; epoch < epochs; epoch++) {
        shuffle_order(order, num_samples);
        for (int position = 0; position < num_samples; position++) {
            int sample = order[position];
            /* These two calls are the complete learning step for one sample. */
            forward_pass(inputs[sample], net);
            backward_pass(targets[sample], net, learning_rate);
        }
    }
    free(order);
    return 0;
}

ClassificationMetrics evaluate_classifier(double **inputs, double **targets,
                                            int num_samples, Net *net,
                                            int *confusion_matrix) {
    /*
     * This loop is the evaluation "meat." It performs forward passes only, so
     * learned weights and biases do not change (the activation workspace does).
     * It reports exact-class accuracy and mean cross-entropy.
     */
    ClassificationMetrics metrics = {0, num_samples, 0.0};
    int output_layer = net->num_layers - 1;
    int num_classes = net->layer_sizes[output_layer];
    if (confusion_matrix) {
        memset(confusion_matrix, 0,
               sizeof(*confusion_matrix) * (size_t)num_classes *
                   (size_t)num_classes);
    }

    for (int sample = 0; sample < num_samples; sample++) {
        forward_pass(inputs[sample], net);
        int predicted = index_of_max(net->activations[output_layer], num_classes);
        int actual = index_of_max(targets[sample], num_classes);
        if (predicted == actual) {
            metrics.correct++;
        }
        if (confusion_matrix) {
            /* Row = actual class, column = predicted class. */
            confusion_matrix[actual * num_classes + predicted]++;
        }
        double probability = net->activations[output_layer][actual];
        /* Cross-entropy for a one-hot target is -log(P(actual class)). */
        metrics.cross_entropy -= log(probability > DBL_MIN ? probability : DBL_MIN);
    }
    if (num_samples > 0) {
        metrics.cross_entropy /= num_samples;
    }
    return metrics;
}

void print_confusion_matrix(const int *matrix, int num_classes) {
    printf("\nConfusion matrix (rows=actual, columns=predicted):\n     ");
    for (int predicted = 0; predicted < num_classes; predicted++) {
        printf(" %4d", predicted);
    }
    printf("\n");
    for (int actual = 0; actual < num_classes; actual++) {
        printf("%4d ", actual);
        for (int predicted = 0; predicted < num_classes; predicted++) {
            printf(" %4d", matrix[actual * num_classes + predicted]);
        }
        printf("\n");
    }
}

/* --- Binary model format ------------------------------------------------ */

/*
 * Files contain, in order and without padding:
 *
 *   uint32 magic, uint32 version, uint32 number_of_layers
 *   int32 layer_size[number_of_layers]
 *   double weights for each adjacent layer pair (source-major matrix order)
 *   double biases for each adjacent layer pair
 *
 * The fixed-width integer fields make their sizes explicit. Doubles and all
 * multi-byte values are written in the host's byte order, so load_net() expects
 * the same binary representation that save_net() wrote. The magic and version
 * reject unrelated or future formats before any large allocation is attempted.
 */

static int write_items(FILE *file, const void *items, size_t item_size,
                       size_t item_count) {
    return fwrite(items, item_size, item_count, file) == item_count ? 0 : -1;
}

static int finite_parameters(const Net *net) {
    /* NaN or infinity would poison every later prediction, so reject either. */
    for (int layer = 0; layer < net->num_layers - 1; layer++) {
        size_t weight_count = (size_t)net->layer_sizes[layer] *
                              (size_t)net->layer_sizes[layer + 1];
        for (size_t index = 0; index < weight_count; index++) {
            if (!isfinite(net->weights[layer][index])) return 0;
        }
        for (int index = 0; index < net->layer_sizes[layer + 1]; index++) {
            if (!isfinite(net->biases[layer][index])) return 0;
        }
    }
    return 1;
}

int save_net(const Net *net, const char *filename) {
    if (!net || !filename || !finite_parameters(net)) {
        fprintf(stderr, "Cannot save a model with invalid parameters\n");
        return -1;
    }
    FILE *file = fopen(filename, "wb");
    if (!file) {
        fprintf(stderr, "Failed to open model '%s' for writing: ", filename);
        perror(NULL);
        return -1;
    }

    /* Header and architecture come first so a reader can size the network. */
    uint32_t magic = MODEL_MAGIC;
    uint32_t version = MODEL_VERSION;
    uint32_t num_layers = (uint32_t)net->num_layers;
    int status = 0;
    if (write_items(file, &magic, sizeof(magic), 1) != 0 ||
        write_items(file, &version, sizeof(version), 1) != 0 ||
        write_items(file, &num_layers, sizeof(num_layers), 1) != 0) {
        status = -1;
    }
    for (int layer = 0; status == 0 && layer < net->num_layers; layer++) {
        int32_t size = (int32_t)net->layer_sizes[layer];
        status = write_items(file, &size, sizeof(size), 1);
    }
    /* Parameters are grouped as all weight matrices, then all bias vectors. */
    for (int layer = 0; status == 0 && layer < net->num_layers - 1; layer++) {
        size_t count = (size_t)net->layer_sizes[layer] *
                       (size_t)net->layer_sizes[layer + 1];
        status = write_items(file, net->weights[layer], sizeof(double), count);
    }
    for (int layer = 0; status == 0 && layer < net->num_layers - 1; layer++) {
        size_t count = (size_t)net->layer_sizes[layer + 1];
        status = write_items(file, net->biases[layer], sizeof(double), count);
    }
    if (fclose(file) != 0) {
        status = -1;
    }
    if (status != 0) {
        /* Do not leave a truncated file looking like a successful save. */
        fprintf(stderr, "Failed while writing model '%s'\n", filename);
        remove(filename);
    }
    return status;
}

Net *load_net(const char *filename) {
    if (!filename) {
        return NULL;
    }
    FILE *file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Failed to open model '%s' for reading: ", filename);
        perror(NULL);
        return NULL;
    }

    /* Validate the small fixed header before trusting any stored dimensions. */
    uint32_t magic = 0;
    uint32_t version = 0;
    uint32_t stored_layers = 0;
    if (fread(&magic, sizeof(magic), 1, file) != 1 || magic != MODEL_MAGIC ||
        fread(&version, sizeof(version), 1, file) != 1 ||
        version != MODEL_VERSION ||
        fread(&stored_layers, sizeof(stored_layers), 1, file) != 1 ||
        stored_layers < 2 || stored_layers > MAX_MODEL_LAYERS) {
        fprintf(stderr, "Invalid or unsupported model '%s'\n", filename);
        fclose(file);
        return NULL;
    }

    int num_layers = (int)stored_layers;
    int *layer_sizes = malloc(sizeof(*layer_sizes) * (size_t)num_layers);
    if (!layer_sizes) {
        perror("Failed to allocate model architecture");
        fclose(file);
        return NULL;
    }
    int status = 0;
    for (int layer = 0; layer < num_layers; layer++) {
        int32_t size = 0;
        if (fread(&size, sizeof(size), 1, file) != 1 || size <= 0 ||
            size > MAX_LAYER_SIZE) {
            status = -1;
            break;
        }
        layer_sizes[layer] = (int)size;
    }

    /* allocate_net() repeats dimension and multiplication-overflow checks. */
    Net *net = NULL;
    if (status == 0) {
        net = allocate_net(layer_sizes, num_layers);
        if (!net) {
            status = -1;
        }
    }
    free(layer_sizes);

    for (int layer = 0; status == 0 && layer < num_layers - 1; layer++) {
        size_t count = (size_t)net->layer_sizes[layer] *
                       (size_t)net->layer_sizes[layer + 1];
        if (fread(net->weights[layer], sizeof(double), count, file) != count) {
            status = -1;
        }
    }
    for (int layer = 0; status == 0 && layer < num_layers - 1; layer++) {
        size_t count = (size_t)net->layer_sizes[layer + 1];
        if (fread(net->biases[layer], sizeof(double), count, file) != count) {
            status = -1;
        }
    }
    /* Reject non-finite parameters and any trailing bytes after the payload. */
    if (status == 0 && !finite_parameters(net)) {
        status = -1;
    }
    if (status == 0 && fgetc(file) != EOF) {
        status = -1;
    }
    if (ferror(file)) {
        status = -1;
    }
    if (fclose(file) != 0) {
        status = -1;
    }
    if (status != 0) {
        fprintf(stderr, "Model '%s' is truncated or malformed\n", filename);
        free_net(net);
        return NULL;
    }
    return net;
}

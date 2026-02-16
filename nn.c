/*
 * nn.c - Neural Network Implementation
 *
 * This file implements a fully-connected feedforward neural network with
 * backpropagation learning. The network supports arbitrary architectures
 * (any number of layers and neurons per layer) configured via config files.
 *
 * Key Features:
 * - Dynamic architecture (configured at runtime)
 * - Xavier weight initialization for better convergence
 * - Stochastic gradient descent with shuffling
 * - OpenMP parallelization for multi-core systems
 * - Model persistence (save/load trained networks)
 * - Multi-class classification support
 *
 * Algorithm:
 * 1. Forward propagation: Compute network output for given input
 * 2. Backward propagation: Compute gradients using chain rule
 * 3. Weight update: Adjust weights to minimize error
 *
 * Educational Implementation - Optimized for clarity and understanding.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "nn.h"
#include "tools.h"

// OpenMP parallelization threshold
// For small layers, thread overhead exceeds benefit
// XOR (2-2-2-1) won't parallelize, but digits (64-128-64-10) will
#define MIN_SIZE_FOR_PARALLEL 100

/*
 * create_net() - Create and initialize a neural network
 *
 * Allocates memory for all network structures (weights, biases, activations, deltas)
 * and initializes weights using Xavier initialization for better convergence.
 *
 * Parameters:
 *   config - Network configuration (layer sizes, learning rate, etc.)
 *
 * Returns:
 *   Pointer to initialized network, or NULL on allocation failure
 *
 * Memory Layout:
 *   - weights[layer][i*next_size + j]: Weight from neuron i to neuron j
 *   - biases[layer][j]: Bias for neuron j in next layer
 *   - activations[layer][i]: Output of neuron i (cached during forward pass)
 *   - deltas[layer][i]: Error gradient for neuron i (used in backprop)
 */
Net* create_net(const Config *config) {
    Net *net = malloc(sizeof(Net));
    if (!net) {
        perror("Failed to allocate network");
        return NULL;
    }

    net->num_layers = config->num_hidden_layers + 2;
    net->layer_sizes = malloc(sizeof(int) * net->num_layers);

    net->layer_sizes[0] = config->input_size;
    for (int i = 0; i < config->num_hidden_layers; i++) {
        net->layer_sizes[i + 1] = config->hidden_layer_sizes[i];
    }
    net->layer_sizes[net->num_layers - 1] = config->output_size;

    net->weights = malloc(sizeof(double*) * (net->num_layers - 1));
    net->biases = malloc(sizeof(double*) * (net->num_layers - 1));
    net->activations = malloc(sizeof(double*) * net->num_layers);
    net->deltas = malloc(sizeof(double*) * net->num_layers);

    for (int layer = 0; layer < net->num_layers; layer++) {
        net->activations[layer] = malloc(sizeof(double) * net->layer_sizes[layer]);
        memset(net->activations[layer], 0, sizeof(double) * net->layer_sizes[layer]);

        net->deltas[layer] = malloc(sizeof(double) * net->layer_sizes[layer]);
        memset(net->deltas[layer], 0, sizeof(double) * net->layer_sizes[layer]);
    }

    for (int layer = 0; layer < net->num_layers - 1; layer++) {
        int current_size = net->layer_sizes[layer];
        int next_size = net->layer_sizes[layer + 1];

        net->weights[layer] = malloc(sizeof(double) * current_size * next_size);
        net->biases[layer] = malloc(sizeof(double) * next_size);

        // Use Xavier initialization for better weight initialization
        // Xavier helps prevent vanishing/exploding gradients in deep networks
        for (int i = 0; i < current_size * next_size; i++) {
            net->weights[layer][i] = xavier_init(current_size, next_size);
        }
        for (int i = 0; i < next_size; i++) {
            // Biases initialized to small random values
            net->biases[layer][i] = random_uniform_init() * 0.01;
        }
    }

    return net;
}

void free_net(Net *net) {
    if (!net) return;

    for (int layer = 0; layer < net->num_layers; layer++) {
        free(net->activations[layer]);
        free(net->deltas[layer]);
    }
    free(net->activations);
    free(net->deltas);

    for (int layer = 0; layer < net->num_layers - 1; layer++) {
        free(net->weights[layer]);
        free(net->biases[layer]);
    }
    free(net->weights);
    free(net->biases);
    free(net->layer_sizes);
    free(net);
}

/*
 * forward_pass() - Compute network output (prediction) for given input
 *
 * Propagates input through the network layer by layer:
 * 1. Copy input to first layer activations
 * 2. For each layer transition:
 *    a. Compute weighted sum: sum = Σ(activation[i] * weight[i→j]) + bias[j]
 *    b. Apply sigmoid activation: activation[j] = 1 / (1 + e^(-sum))
 * 3. Final layer contains the network's output/prediction
 *
 * Parameters:
 *   inputs - Input vector (length = layer_sizes[0])
 *   net    - Neural network structure (activations will be updated)
 *
 * Result:
 *   net->activations[num_layers-1] contains the network's output
 */
void forward_pass(const double *inputs, Net *net) {
    // Copy input values into first layer (input layer just holds the data)
    memcpy(net->activations[0], inputs, sizeof(double) * net->layer_sizes[0]);

    // Propagate through each layer
    for (int layer = 0; layer < net->num_layers - 1; layer++) {
        int current_size = net->layer_sizes[layer];
        int next_size = net->layer_sizes[layer + 1];

        // Only use OpenMP for larger layers (avoids thread overhead on small networks)
        #pragma omp parallel for schedule(static) if(next_size > MIN_SIZE_FOR_PARALLEL)
        for (int j = 0; j < next_size; j++) {
            double sum = net->biases[layer][j];
            for (int i = 0; i < current_size; i++) {
                sum += net->activations[layer][i] * net->weights[layer][i * next_size + j];
            }
            net->activations[layer + 1][j] = sigmoid(sum);
        }
    }
}

/*
 * backward_pass() - Backpropagation: Compute gradients and update weights
 *
 * Implements the backpropagation algorithm to train the network:
 *
 * Phase 1 - Output Error:
 *   For each output neuron:
 *     error = expected - actual
 *     delta = error * sigmoid'(activation)
 *
 * Phase 2 - Hidden Layer Error (chain rule):
 *   For each hidden layer (backward):
 *     For each neuron:
 *       error = Σ(delta[next] * weight[current→next])
 *       delta = error * sigmoid'(activation)
 *
 * Phase 3 - Weight Update (gradient descent):
 *   For each weight:
 *     weight += learning_rate * delta[next] * activation[current]
 *   For each bias:
 *     bias += learning_rate * delta
 *
 * Parameters:
 *   inputs        - Input vector (not used in this implementation)
 *   expected      - Target output vector
 *   net           - Neural network (weights/biases will be updated)
 *   learning_rate - Step size for gradient descent (typically 0.1-0.5)
 */
void backward_pass(const double *inputs, const double *expected, Net *net, double learning_rate) {
    int num_layers = net->num_layers;
    double **deltas = net->deltas;  // Error gradients (pre-allocated)

    // Clear deltas from previous iteration
    for (int layer = 0; layer < num_layers; layer++) {
        memset(deltas[layer], 0, sizeof(double) * net->layer_sizes[layer]);
    }

    // Phase 1: Compute output layer error
    int output_layer = num_layers - 1;
    int output_size = net->layer_sizes[output_layer];
    #pragma omp parallel for schedule(static) if(output_size > MIN_SIZE_FOR_PARALLEL)
    for (int i = 0; i < output_size; i++) {
        double error = expected[i] - net->activations[output_layer][i];
        // delta = error * sigmoid'(activation)
        // The sigmoid derivative tells us how sensitive this neuron is to changes
        deltas[output_layer][i] = error * sigmoid_derivative(net->activations[output_layer][i]);
    }

    // Phase 2: Propagate error backward through hidden layers
    for (int layer = num_layers - 2; layer >= 0; layer--) {
        int current_size = net->layer_sizes[layer];
        int next_size = net->layer_sizes[layer + 1];

        #pragma omp parallel for schedule(static) if(current_size > MIN_SIZE_FOR_PARALLEL)
        for (int i = 0; i < current_size; i++) {
            double error = 0.0;
            // Accumulate weighted error from next layer (chain rule)
            for (int j = 0; j < next_size; j++) {
                error += deltas[layer + 1][j] * net->weights[layer][i * next_size + j];
            }
            // Multiply by derivative to get gradient for this neuron
            deltas[layer][i] = error * sigmoid_derivative(net->activations[layer][i]);
        }
    }

    // Phase 3: Update weights and biases using computed gradients
    for (int layer = 0; layer < num_layers - 1; layer++) {
        int current_size = net->layer_sizes[layer];
        int next_size = net->layer_sizes[layer + 1];

        #pragma omp parallel for schedule(static) if(current_size > MIN_SIZE_FOR_PARALLEL)
        for (int i = 0; i < current_size; i++) {
            for (int j = 0; j < next_size; j++) {
                // Update rule: weight += learning_rate * gradient * input
                // If delta is positive, increase weight; if negative, decrease
                // If activation is high, this neuron strongly influences the output
                net->weights[layer][i * next_size + j] +=
                    learning_rate * deltas[layer + 1][j] * net->activations[layer][i];
            }
        }

        // Update biases (simpler: no activation multiplication needed)
        #pragma omp parallel for schedule(static) if(next_size > MIN_SIZE_FOR_PARALLEL)
        for (int j = 0; j < next_size; j++) {
            net->biases[layer][j] += learning_rate * deltas[layer + 1][j];
        }
    }
    // Note: deltas are pre-allocated in Net struct, so no need to free
}

/*
 * train_nn() - Train network using stochastic gradient descent
 *
 * Implements online learning with shuffling:
 * - Each epoch processes all training samples in random order
 * - Shuffling prevents the network from memorizing sequence patterns
 * - Weights are updated after each sample (stochastic gradient descent)
 *
 * Parameters:
 *   inputs        - Array of input vectors [num_samples][input_size]
 *   expected      - Array of expected outputs [num_samples]
 *   num_samples   - Number of training examples
 *   net           - Neural network to train
 *   rounds        - Number of epochs (full passes through dataset)
 *   learning_rate - Step size for weight updates
 */
void train_nn(double **inputs, double *expected, int num_samples, Net *net, int rounds, double learning_rate) {
    int *order = NULL;

    for (int round = 0; round < rounds; round++) {
        // Shuffle training order each epoch
        order = init_order_array(num_samples);

        // Train on each sample in random order
        for(int i = 0; i < num_samples; ++i) {
            forward_pass(inputs[order[i]], net);
            backward_pass(inputs[order[i]], &expected[order[i]], net, learning_rate);
        }

        free(order);
    }
}

void test_nn(double **inputs, double *expected, int num_samples, Net *net) {
    printf("\n=== What did we learn? ===\n");
    for(int i = 0; i < num_samples; i++) {
        forward_pass(inputs[i], net);
        int output_layer = net->num_layers - 1;
        printf("[");
        for (int j = 0; j < net->layer_sizes[0]; j++) {
            printf("%.0f", inputs[i][j]);
            if (j < net->layer_sizes[0] - 1) printf(",");
        }
        printf("] → %.3f (want %.0f)\n",
            net->activations[output_layer][0], expected[i]);
    }
}

double test_nn_and_get_mse(double **inputs, double *expected, int num_samples, Net *net) {
    double mse = 0.0;
    for(int i = 0; i < num_samples; i++) {
        forward_pass(inputs[i], net);
        int output_layer = net->num_layers - 1;
        mse += pow(expected[i] - net->activations[output_layer][0], 2);
    }
    return mse / num_samples;
}

int save_net(const Net *net, const char *filename) {
    FILE *f = fopen(filename, "wb");
    if (!f) {
        perror("Failed to open file for writing");
        return -1;
    }

    // Write magic number and version
    uint32_t magic = 0x4E454E45;  // "ENEN" in little-endian
    uint32_t version = 1;
    fwrite(&magic, sizeof(uint32_t), 1, f);
    fwrite(&version, sizeof(uint32_t), 1, f);

    // Write network architecture
    uint32_t num_layers = (uint32_t)net->num_layers;
    fwrite(&num_layers, sizeof(uint32_t), 1, f);

    for (int i = 0; i < net->num_layers; i++) {
        int32_t size = (int32_t)net->layer_sizes[i];
        fwrite(&size, sizeof(int32_t), 1, f);
    }

    // Write weights
    for (int layer = 0; layer < net->num_layers - 1; layer++) {
        int current_size = net->layer_sizes[layer];
        int next_size = net->layer_sizes[layer + 1];
        int weight_count = current_size * next_size;
        fwrite(net->weights[layer], sizeof(double), weight_count, f);
    }

    // Write biases
    for (int layer = 0; layer < net->num_layers - 1; layer++) {
        int next_size = net->layer_sizes[layer + 1];
        fwrite(net->biases[layer], sizeof(double), next_size, f);
    }

    fclose(f);
    return 0;
}

Net* load_net(const char *filename) {
    FILE *f = fopen(filename, "rb");
    if (!f) {
        perror("Failed to open file for reading");
        return NULL;
    }

    // Read and verify magic number
    uint32_t magic, version;
    if (fread(&magic, sizeof(uint32_t), 1, f) != 1 || magic != 0x4E454E45) {
        fprintf(stderr, "Invalid model file format (expected ENEN magic)\n");
        fclose(f);
        return NULL;
    }

    // Read version
    if (fread(&version, sizeof(uint32_t), 1, f) != 1 || version != 1) {
        fprintf(stderr, "Unsupported model file version\n");
        fclose(f);
        return NULL;
    }

    // Read network architecture
    uint32_t num_layers;
    if (fread(&num_layers, sizeof(uint32_t), 1, f) != 1) {
        fprintf(stderr, "Failed to read number of layers\n");
        fclose(f);
        return NULL;
    }

    Net *net = malloc(sizeof(Net));
    if (!net) {
        perror("Failed to allocate network");
        fclose(f);
        return NULL;
    }

    net->num_layers = (int)num_layers;
    net->layer_sizes = malloc(sizeof(int) * net->num_layers);

    for (int i = 0; i < net->num_layers; i++) {
        int32_t size;
        if (fread(&size, sizeof(int32_t), 1, f) != 1) {
            fprintf(stderr, "Failed to read layer size\n");
            free(net->layer_sizes);
            free(net);
            fclose(f);
            return NULL;
        }
        net->layer_sizes[i] = (int)size;
    }

    // Allocate arrays
    net->weights = malloc(sizeof(double*) * (net->num_layers - 1));
    net->biases = malloc(sizeof(double*) * (net->num_layers - 1));
    net->activations = malloc(sizeof(double*) * net->num_layers);
    net->deltas = malloc(sizeof(double*) * net->num_layers);

    for (int layer = 0; layer < net->num_layers; layer++) {
        net->activations[layer] = malloc(sizeof(double) * net->layer_sizes[layer]);
        memset(net->activations[layer], 0, sizeof(double) * net->layer_sizes[layer]);

        net->deltas[layer] = malloc(sizeof(double) * net->layer_sizes[layer]);
        memset(net->deltas[layer], 0, sizeof(double) * net->layer_sizes[layer]);
    }

    // Read weights
    for (int layer = 0; layer < net->num_layers - 1; layer++) {
        int current_size = net->layer_sizes[layer];
        int next_size = net->layer_sizes[layer + 1];
        int weight_count = current_size * next_size;

        net->weights[layer] = malloc(sizeof(double) * weight_count);
        if (fread(net->weights[layer], sizeof(double), weight_count, f) != (size_t)weight_count) {
            fprintf(stderr, "Failed to read weights for layer %d\n", layer);
            free_net(net);
            fclose(f);
            return NULL;
        }
    }

    // Read biases
    for (int layer = 0; layer < net->num_layers - 1; layer++) {
        int next_size = net->layer_sizes[layer + 1];
        net->biases[layer] = malloc(sizeof(double) * next_size);
        if (fread(net->biases[layer], sizeof(double), next_size, f) != (size_t)next_size) {
            fprintf(stderr, "Failed to read biases for layer %d\n", layer);
            free_net(net);
            fclose(f);
            return NULL;
        }
    }

    fclose(f);
    return net;
}

void train_nn_multiclass(double **inputs, double **expected, int num_samples, Net *net, int rounds, double learning_rate) {
    int *order = NULL;

    for (int round = 0; round < rounds; round++) {
        order = init_order_array(num_samples);

        for(int i = 0; i < num_samples; ++i) {
            forward_pass(inputs[order[i]], net);
            backward_pass(inputs[order[i]], expected[order[i]], net, learning_rate);
        }

        free(order);
    }
}

void test_nn_multiclass(double **inputs, double **expected, int num_samples, Net *net) {
    printf("\n=== Classification Results ===\n");
    int correct = 0;

    for(int i = 0; i < num_samples; i++) {
        forward_pass(inputs[i], net);
        int output_layer = net->num_layers - 1;
        int output_size = net->layer_sizes[output_layer];

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

        int is_correct = (predicted == expected_class);
        if (is_correct) correct++;

        printf("Sample %d: predicted=%d, expected=%d [%s]\n",
               i, predicted, expected_class, is_correct ? "OK" : "FAIL");
    }

    printf("\nAccuracy: %d/%d (%.1f%%)\n", correct, num_samples,
           100.0 * correct / num_samples);
}

double test_nn_and_get_mse_multiclass(double **inputs, double **expected, int num_samples, Net *net) {
    double mse = 0.0;
    for(int i = 0; i < num_samples; i++) {
        forward_pass(inputs[i], net);
        int output_layer = net->num_layers - 1;
        int output_size = net->layer_sizes[output_layer];

        for (int j = 0; j < output_size; j++) {
            double error = expected[i][j] - net->activations[output_layer][j];
            mse += error * error;
        }
    }
    return mse / num_samples;
}

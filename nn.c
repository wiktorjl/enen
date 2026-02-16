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

    for (int layer = 0; layer < net->num_layers; layer++) {
        net->activations[layer] = malloc(sizeof(double) * net->layer_sizes[layer]);
        memset(net->activations[layer], 0, sizeof(double) * net->layer_sizes[layer]);
    }

    for (int layer = 0; layer < net->num_layers - 1; layer++) {
        int current_size = net->layer_sizes[layer];
        int next_size = net->layer_sizes[layer + 1];

        net->weights[layer] = malloc(sizeof(double) * current_size * next_size);
        net->biases[layer] = malloc(sizeof(double) * next_size);

        for (int i = 0; i < current_size * next_size; i++) {
            net->weights[layer][i] = randinit();
        }
        for (int i = 0; i < next_size; i++) {
            net->biases[layer][i] = randinit();
        }
    }

    return net;
}

void free_net(Net *net) {
    if (!net) return;

    for (int layer = 0; layer < net->num_layers; layer++) {
        free(net->activations[layer]);
    }
    free(net->activations);

    for (int layer = 0; layer < net->num_layers - 1; layer++) {
        free(net->weights[layer]);
        free(net->biases[layer]);
    }
    free(net->weights);
    free(net->biases);
    free(net->layer_sizes);
    free(net);
}

void forward_pass(const double *inputs, Net *net) {
    memcpy(net->activations[0], inputs, sizeof(double) * net->layer_sizes[0]);

    for (int layer = 0; layer < net->num_layers - 1; layer++) {
        int current_size = net->layer_sizes[layer];
        int next_size = net->layer_sizes[layer + 1];

        #pragma omp parallel for schedule(static)
        for (int j = 0; j < next_size; j++) {
            double sum = net->biases[layer][j];
            for (int i = 0; i < current_size; i++) {
                sum += net->activations[layer][i] * net->weights[layer][i * next_size + j];
            }
            net->activations[layer + 1][j] = sigmoid(sum);
        }
    }
}

void backward_pass(const double *inputs, const double *expected, Net *net, double learning_rate) {
    int num_layers = net->num_layers;

    double **deltas = malloc(sizeof(double*) * num_layers);
    for (int layer = 0; layer < num_layers; layer++) {
        deltas[layer] = calloc(net->layer_sizes[layer], sizeof(double));
    }

    int output_layer = num_layers - 1;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < net->layer_sizes[output_layer]; i++) {
        double error = expected[i] - net->activations[output_layer][i];
        deltas[output_layer][i] = error * sigmoid_derivative(net->activations[output_layer][i]);
    }

    for (int layer = num_layers - 2; layer >= 0; layer--) {
        int current_size = net->layer_sizes[layer];
        int next_size = net->layer_sizes[layer + 1];

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < current_size; i++) {
            double error = 0.0;
            for (int j = 0; j < next_size; j++) {
                error += deltas[layer + 1][j] * net->weights[layer][i * next_size + j];
            }
            deltas[layer][i] = error * sigmoid_derivative(net->activations[layer][i]);
        }
    }

    for (int layer = 0; layer < num_layers - 1; layer++) {
        int current_size = net->layer_sizes[layer];
        int next_size = net->layer_sizes[layer + 1];

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < current_size; i++) {
            for (int j = 0; j < next_size; j++) {
                net->weights[layer][i * next_size + j] += learning_rate * deltas[layer + 1][j] * net->activations[layer][i];
            }
        }

        #pragma omp parallel for schedule(static)
        for (int j = 0; j < next_size; j++) {
            net->biases[layer][j] += learning_rate * deltas[layer + 1][j];
        }
    }

    for (int layer = 0; layer < num_layers; layer++) {
        free(deltas[layer]);
    }
    free(deltas);
}

void train_nn(double **inputs, double *expected, int num_samples, Net *net, int rounds, double learning_rate) {
    int *order = NULL;

    for (int round = 0; round < rounds; round++) {
        order = init_order_array(num_samples);

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

    for (int layer = 0; layer < net->num_layers; layer++) {
        net->activations[layer] = malloc(sizeof(double) * net->layer_sizes[layer]);
        memset(net->activations[layer], 0, sizeof(double) * net->layer_sizes[layer]);
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

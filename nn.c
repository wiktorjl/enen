#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    for (int i = 0; i < net->layer_sizes[output_layer]; i++) {
        double error = expected[i] - net->activations[output_layer][i];
        deltas[output_layer][i] = error * sigmoid_derivative(net->activations[output_layer][i]);
    }

    for (int layer = num_layers - 2; layer >= 0; layer--) {
        int current_size = net->layer_sizes[layer];
        int next_size = net->layer_sizes[layer + 1];

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

        for (int i = 0; i < current_size; i++) {
            for (int j = 0; j < next_size; j++) {
                net->weights[layer][i * next_size + j] += learning_rate * deltas[layer + 1][j] * net->activations[layer][i];
            }
        }

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

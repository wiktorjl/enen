#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <ctype.h>

#include "tools.h"

void print_net(const Net *net, int verbose) {
    if (!net) {
        printf("Net is NULL\n");
        return;
    }

    printf("Network Architecture: ");
    for (int i = 0; i < net->num_layers; i++) {
        printf("%d", net->layer_sizes[i]);
        if (i < net->num_layers - 1) printf("-");
    }
    printf("\n");

    if (verbose) {
        for (int layer = 0; layer < net->num_layers - 1; layer++) {
            int current_size = net->layer_sizes[layer];
            int next_size = net->layer_sizes[layer + 1];

            printf("\nLayer %d -> %d:\n", layer, layer + 1);
            printf("Weights:\n");
            for (int i = 0; i < current_size; i++) {
                for (int j = 0; j < next_size; j++) {
                    printf("  [%d][%d]: %f\n", i, j, net->weights[layer][i * next_size + j]);
                }
            }
            printf("Biases:\n");
            for (int j = 0; j < next_size; j++) {
                printf("  [%d]: %f\n", j, net->biases[layer][j]);
            }
        }
    }

    printf("\nActivations:\n");
    for (int layer = 0; layer < net->num_layers; layer++) {
        printf("Layer %d:\n", layer);
        for (int i = 0; i < net->layer_sizes[layer]; i++) {
            printf("  [%d]: %f\n", i, net->activations[layer][i]);
        }
    }
}

double randinit() {
    return 2.0 * ((double)rand() / (double)RAND_MAX) - 1.0;
}

double xavier_init(int fan_in, int fan_out) {
    double limit = sqrt(6.0 / (fan_in + fan_out));
    return limit * (2.0 * ((double)rand() / (double)RAND_MAX) - 1.0);
}

double sigmoid(double input) {
    return 1.0 / (1.0 + exp(-input));
}

double sigmoid_derivative(double sigmoid_output) {
    return sigmoid_output * (1.0 - sigmoid_output);
}

void shuffle_array(int n, double *arr) {
    for (int i = n - 1; i > 0; --i) {
        int j = rand() % (i + 1);
        double tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }
}

void load_dataset(const char *filename, double ***inputs_out, double **expected_out, int *num_samples_out, int input_size) {
    FILE *f = fopen(filename, "r");
    if (!f) {
        perror("Failed to open dataset file");
        exit(1);
    }

    int capacity = 10;
    int count = 0;
    double **inputs = malloc(sizeof(double*) * capacity);
    double *expected = malloc(sizeof(double) * capacity);

    char line[1024];
    while (fgets(line, sizeof(line), f)) {
        if (count >= capacity) {
            capacity *= 2;
            inputs = realloc(inputs, sizeof(double*) * capacity);
            expected = realloc(expected, sizeof(double) * capacity);
        }

        inputs[count] = malloc(sizeof(double) * input_size);
        char *token = strtok(line, ",");
        int col = 0;
        while (token && col < input_size) {
            inputs[count][col] = atof(token);
            token = strtok(NULL, ",");
            col++;
        }
        if (token) {
            expected[count] = atof(token);
        }
        count++;
    }

    fclose(f);

    *inputs_out = inputs;
    *expected_out = expected;
    *num_samples_out = count;
}

void free_dataset(double **inputs, double *expected, int num_samples) {
    for (int i = 0; i < num_samples; i++) {
        free(inputs[i]);
    }
    free(inputs);
    free(expected);
}

int *init_order_array(int n) {
    int *arr = (int *)malloc(n * sizeof(int));
    if (!arr) return NULL;
    for (int i = 0; i < n; ++i) {
        arr[i] = i;
    }
    // Shuffle the array
    for (int i = n - 1; i > 0; --i) {
        int j = rand() % (i + 1);
        int tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }
    return arr;
}

char *trim_copy(char *src, char *dest, int destsize) {
    while (isspace(*src)) src++;
    char *end = src + strlen(src) - 1;
    while (end > src && isspace(*end)) end--;

    int len = end - src + 1;
    if (len >= destsize) len = destsize - 1;
    memcpy(dest, src, len);
    dest[len] = '\0';
    return dest;
}
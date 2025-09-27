#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "tools.h"

void print_net(const Net *net, int verbose) {
    int i, j;

    if(verbose) {
        printf("weights_ih:\n");
        for (i = 0; i < 2; ++i) {
            for (j = 0; j < 2; ++j) {
                printf("  [%d][%d]: %f\n", i, j, net->weights_ih[i][j]);
            }
        }
        printf("weights_ho:\n");
        for (i = 0; i < 2; ++i) {
            for (j = 0; j < 1; ++j) {
                printf("  [%d][%d]: %f\n", i, j, net->weights_ho[i][j]);
            }
        }
        printf("bias_hidden:\n");
        for (i = 0; i < 2; ++i) {
            printf("  [%d]: %f\n", i, net->bias_hidden[i]);
        }
        printf("bias_output:\n");
        for (i = 0; i < 1; ++i) {
            printf("  [%d]: %f\n", i, net->bias_output[i]);
        }
    }


    printf("output_hidden:\n");
    for (i = 0; i < 2; ++i) {
        printf("  [%d]: %f\n", i, net->output_hidden[i]);
    }
    printf("output_final:\n");
    for (i = 0; i < 1; ++i) {
        printf("  [%d]: %f\n", i, net->output_final[i]);
    }
}

double randinit() {
    return 2.0 * ((double)rand() / (double)RAND_MAX) - 1.0;
}

void init_net(Net *net) {
    for (int i = 0; i < 2; ++i) {
        net->bias_hidden[i] = randinit();
    }
    for (int i = 0; i < 1; ++i) {
        net->bias_output[i] = randinit();
    }

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            net->weights_ih[i][j] = randinit();
        }
    }
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 1; ++j) {
            net->weights_ho[i][j] = randinit();
        }
    }


    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 1; ++j) {
            net->output_hidden[i] = 0;
        }
    }
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

void init_xor_data(const char *filename, double inputs[4][2], double expected[4]) {
    FILE *f = fopen(filename, "r");
    if (!f) {
        perror("Failed to open dataset file");
        exit(1);
    }
    for (int i = 0; i < 4; ++i) {
        if (fscanf(f, "%lf,%lf,%lf", &inputs[i][0], &inputs[i][1], &expected[i]) != 3) {
            fprintf(stderr, "Error reading line %d from dataset file\n", i);
            fclose(f);
            exit(1);
        }
    }
    fclose(f);
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
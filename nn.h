#ifndef NN_H
#define NN_H

#include "config.h"

typedef struct NetStruct {
    int num_layers;
    int *layer_sizes;

    double **weights;
    double **biases;
    double **activations;
} Net;

Net* create_net(const Config *config);
void free_net(Net *net);

void forward_pass(const double *inputs, Net *net);
void backward_pass(const double *inputs, const double *expected, Net *net, double learning_rate);
void train_nn(double **inputs, double *expected, int num_samples, Net *net, int rounds, double learning_rate);
void test_nn(double **inputs, double *expected, int num_samples, Net *net);
double test_nn_and_get_mse(double **inputs, double *expected, int num_samples, Net *net);

#endif // NN_H

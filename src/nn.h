#ifndef NN_H
#define NN_H

#include "config.h"

typedef struct NetStruct {
    int num_layers;
    int *layer_sizes;

    double **weights;
    double **biases;
    double **activations;
    double **deltas;        // Error gradients for backpropagation (pre-allocated)
} Net;

Net* create_net(const Config *config);
void free_net(Net *net);

void forward_pass(const double *inputs, Net *net);
void backward_pass(const double *inputs, const double *expected, Net *net, double learning_rate);
void train_nn(double **inputs, double *expected, int num_samples, Net *net, int rounds, double learning_rate);
void test_nn(double **inputs, double *expected, int num_samples, Net *net);
double test_nn_and_get_mse(double **inputs, double *expected, int num_samples, Net *net);

// Model serialization
int save_net(const Net *net, const char *filename);
Net* load_net(const char *filename);

// Multi-class classification functions
void train_nn_multiclass(double **inputs, double **expected, int num_samples, Net *net, int rounds, double learning_rate);
void test_nn_multiclass(double **inputs, double **expected, int num_samples, Net *net);
double test_nn_and_get_mse_multiclass(double **inputs, double **expected, int num_samples, Net *net);

#endif // NN_H

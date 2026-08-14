#ifndef TOOLS_H
#define TOOLS_H

#include "nn.h"

void print_net(const Net *net, int verbose);
double random_uniform_init();  // Random initialization in range [-1, 1]
double xavier_init(int fan_in, int fan_out);  // Xavier/Glorot initialization
double sigmoid(double input);
double sigmoid_derivative(double sigmoid_output);

void shuffle_array(int n, double *arr);
int * init_order_array(int n);

// Multi-class dataset loading
void load_dataset_multiclass(const char *filename, double ***inputs_out, double ***expected_out, int *num_samples_out, int input_size, int num_classes);
void free_dataset_multiclass(double **inputs, double **expected, int num_samples);

char *trim_copy(char *src, char *dest, int destsize);
#endif // TOOLS_H

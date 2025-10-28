#ifndef TOOLS_H
#define TOOLS_H

#include "nn.h"

void print_net(const Net *net, int verbose);
double randinit();
double xavier_init(int fan_in, int fan_out);
double sigmoid(double input);
double sigmoid_derivative(double sigmoid_output);

void shuffle_array(int n, double *arr);
void load_dataset(const char *filename, double ***inputs_out, double **expected_out, int *num_samples_out, int input_size);
void free_dataset(double **inputs, double *expected, int num_samples);
int * init_order_array(int n);

char *trim_copy(char *src, char *dest, int destsize);
#endif // TOOLS_H
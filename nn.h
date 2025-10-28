#ifndef NN_H
#define NN_H

#include "tools.h"

void forward_pass(double x, double y, Net *net);
double backward_pass(double x, double y, double expected, Net *net, double learning_rate);
void train_nn(double inputs[4][2], double expected[4], Net *net, int rounds, double learning_rate);
void test_nn(double inputs[4][2], double expected[4], Net *net);
double test_nn_and_get_mse(double inputs[4][2], double expected[4], Net *net);

#endif // NN_H

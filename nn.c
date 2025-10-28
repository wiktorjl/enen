#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "nn.h"


void forward_pass(double x, double y, Net *net) {

    // CALCULATE HIDDEN LAYER
    net->output_hidden[0] = sigmoid(x * net->weights_ih[0][0] + y * net->weights_ih[0][1] + net->bias_hidden[0]);
    net->output_hidden[1] = sigmoid(x * net->weights_ih[1][0] + y * net->weights_ih[1][1] + net->bias_hidden[1]);

    // CALCULATE OUTPUT LAYER
    net->output_final[0] = sigmoid(net->output_hidden[0] * net->weights_ho[0][0] + net->output_hidden[1] * net->weights_ho[1][0] + net->bias_output[0]);
}

double backward_pass(double x, double y, double expected, Net *net, double learning_rate) {
    double error_output = (expected - net->output_final[0]) * sigmoid_derivative(net->output_final[0]);

    net->weights_ho[0][0] = net->weights_ho[0][0] + learning_rate  * error_output * net->output_hidden[0];
    net->weights_ho[1][0] = net->weights_ho[1][0] + learning_rate  * error_output * net->output_hidden[1];
    net->bias_output[0] = net->bias_output[0] + learning_rate * error_output;

    double error_hidden[2];
    for (int j = 0; j < 2; ++j) {
        error_hidden[j] = error_output * net->weights_ho[j][0] * sigmoid_derivative(net->output_hidden[j]);
    }


    net->weights_ih[0][0] = net->weights_ih[0][0] + learning_rate * error_hidden[0] * x;
    net->weights_ih[0][1] = net->weights_ih[0][1] + learning_rate * error_hidden[0] * y;
    net->weights_ih[1][0] = net->weights_ih[1][0] + learning_rate * error_hidden[1] * x;
    net->weights_ih[1][1] = net->weights_ih[1][1] + learning_rate * error_hidden[1] * y;
    net->bias_hidden[0] = net->bias_hidden[0] + learning_rate * error_hidden[0];
    net->bias_hidden[1] = net->bias_hidden[1] + learning_rate * error_hidden[1];

    return error_output;
}

void train_nn(double inputs[4][2], double expected[4], Net *net, int rounds, double learning_rate) {
    int *order = NULL;

    // TRAIN THE NETWORK
    // For each round, we shuffle the order of the inputs to avoid bias
    // Then we do a forward pass and a backward pass for each input
    for (int round = 0; round < rounds; round++) {
        order = init_order_array(4); // Should return [0,1,2,3] in some random order

        for(int i = 0; i < 4; ++i) {

            // Perform forward pass
            forward_pass(inputs[order[i]][0], inputs[order[i]][1], net);
            // Perform backward pass and get error
            backward_pass(inputs[order[i]][0], inputs[order[i]][1], expected[order[i]], net, learning_rate);

            // printf("Round %d, Input: [%f, %f], Expected: %f, Output: %f, Error: %f, Initial Error: %f\n", round, inputs[order[i]][0], inputs[order[i]][1], expected[order[i]], net->output_final[0], errors[round], errors[0]);
        }

        free(order);
    }
}

void test_nn(double inputs[4][2], double expected[4], Net *net) {
    // TEST THE NETWORK
    printf("\n=== What did we learn? ===\n");
    for(int i = 0; i < 4; i++) {
        forward_pass(inputs[i][0], inputs[i][1], net);
        printf("[%.0f,%.0f] → %.3f (want %.0f)\n", 
            inputs[i][0], inputs[i][1], net->output_final[0], expected[i]);
    }
}

double test_nn_and_get_mse(double inputs[4][2], double expected[4], Net *net) {
    double mse = 0.0;
    for(int i = 0; i < 4; i++) {
        forward_pass(inputs[i][0], inputs[i][1], net);
        mse += pow(expected[i] - net->output_final[0], 2);
    }
    return mse / 4.0;
}
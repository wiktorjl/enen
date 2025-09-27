#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "tools.h"


void forward_pass(double x, double y, Net *net) {

    // CALCULATE HIDDEN LAYER
    double xv = x * net->weights_ih[0][0] + y * net->weights_ih[0][1] + net->bias_hidden[0];
    double yv = x * net->weights_ih[1][0] + y * net->weights_ih[1][1] + net->bias_hidden[1];

    xv = sigmoid(xv);
    yv = sigmoid(yv);

    net->output_hidden[0] = xv;
    net->output_hidden[1] = yv;
    


    // CALCULATE OUTPUT LAYER
    double ov = net->output_hidden[0] * net->weights_ho[0][0] + net->output_hidden[1] * net->weights_ho[1][0] + net->bias_output[0];

    ov = sigmoid(ov);

    net->output_final[0] = ov;
}

double backward_pass(double x, double y, double expected, Net *net) {
    double learning_rate = 0.5;
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

void train_nn(double inputs[4][2], double expected[4], Net *net, double errors[10000]) {
    int *order = NULL;

    // TRAIN THE NETWORK
    // For each round, we shuffle the order of the inputs to avoid bias
    // Then we do a forward pass and a backward pass for each input
    for (int round = 0; round < ROUNDS; round++) {
        order = init_order_array(4); // Should return [0,1,2,3] in some random order

        for(int i = 0; i < 4; ++i) {

            // Perform forward pass
            forward_pass(inputs[order[i]][0], inputs[order[i]][1], net);
            // Perform backward pass and get error
            errors[round] = backward_pass(inputs[order[i]][0], inputs[order[i]][1], expected[order[i]], net);

            // printf("Round %d, Input: [%f, %f], Expected: %f, Output: %f, Error: %f, Initial Error: %f\n", round, inputs[order[i]][0], inputs[order[i]][1], expected[order[i]], net->output_final[0], errors[round], errors[0]);
        }
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

int main() {
    Net * net = malloc(sizeof(Net));
    double inputs[4][2];
    double expected[4];
    double errors[10000];


    srand(time(NULL)); // Seed the random number generator
    init_net(net); // Initialize the neural network memory
    init_xor_data("xor_dataset.csv", inputs, expected); // Grab the XOR data from file

    train_nn(inputs, expected, net, errors);
    test_nn(inputs, expected, net);

    return 0;
}
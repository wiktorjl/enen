#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "nn.h"
#include "config.h"


int main() {
    Net * net = malloc(sizeof(Net));
    double inputs[4][2];
    double expected[4];

    Config* config = load_config("xornet.conf");
    if(!config) {
        fprintf(stderr, "Failed to load config file.\n");
        return 1;
    }

    print_config(config);

    srand(time(NULL)); // Seed the random number generator
    init_net(net); // Initialize the neural network memory
    init_xor_data(config->dataset_path, inputs, expected); // Grab the XOR data from file

    train_nn(inputs, expected, net, config->epochs, config->learning_rate);
    test_nn(inputs, expected, net);

    free(net);
    free_config(config);
    return 0;
}

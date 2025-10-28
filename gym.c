#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "nn.h"

int main() {
    double learning_rates[] = {0.01, 0.1, 0.5, 1.0};
    int rounds[] = {1000, 5000, 10000, 20000};
    int num_learning_rates = sizeof(learning_rates) / sizeof(double);
    int num_rounds = sizeof(rounds) / sizeof(int);

    double inputs[4][2];
    double expected[4];

    init_xor_data("xor_dataset.csv", inputs, expected);

    printf("| Learning Rate | Rounds | MSE      |\n");
    printf("|---------------|--------|----------|\n");

    for (int i = 0; i < num_learning_rates; i++) {
        for (int j = 0; j < num_rounds; j++) {
            Net *net = malloc(sizeof(Net));
            srand(time(NULL));
            init_net(net);

            train_nn(inputs, expected, net, rounds[j], learning_rates[i]);
            double mse = test_nn_and_get_mse(inputs, expected, net);

            printf("| %-13.2f | %-6d | %-8.6f |\n", learning_rates[i], rounds[j], mse);

            free(net);
        }
    }

    return 0;
}

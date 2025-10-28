#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "nn.h"
#include "config.h"
#include "tools.h"

int main() {
    double learning_rates[] = {0.01, 0.1, 0.5, 1.0};
    int rounds[] = {1000, 5000, 10000, 20000};
    int num_learning_rates = sizeof(learning_rates) / sizeof(double);
    int num_rounds = sizeof(rounds) / sizeof(int);

    Config* config = load_config("xornet.conf");
    if(!config) {
        fprintf(stderr, "Failed to load config file.\n");
        return 1;
    }

    double **inputs = NULL;
    double *expected = NULL;
    int num_samples = 0;

    load_dataset(config->dataset_path, &inputs, &expected, &num_samples, config->input_size);

    printf("| Learning Rate | Rounds | MSE      |\n");
    printf("|---------------|--------|----------|\n");

    for (int i = 0; i < num_learning_rates; i++) {
        for (int j = 0; j < num_rounds; j++) {
            srand(time(NULL));
            Net *net = create_net(config);
            if (!net) {
                fprintf(stderr, "Failed to create network.\n");
                continue;
            }

            train_nn(inputs, expected, num_samples, net, rounds[j], learning_rates[i]);
            double mse = test_nn_and_get_mse(inputs, expected, num_samples, net);

            printf("| %-13.2f | %-6d | %-8.6f |\n", learning_rates[i], rounds[j], mse);

            free_net(net);
        }
    }

    free_dataset(inputs, expected, num_samples);
    free_config(config);

    return 0;
}

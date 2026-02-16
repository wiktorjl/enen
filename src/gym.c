#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "nn.h"
#include "config.h"
#include "tools.h"

void print_help(const char *program_name) {
    printf("Usage: %s [OPTIONS]\n\n", program_name);
    printf("Hyperparameter Tuning Tool\n\n");
    printf("Options:\n");
    printf("  --load MODEL    Load pre-trained model and fine-tune with different hyperparameters\n");
    printf("  --help          Display this help message\n\n");
    printf("Default behavior (no options):\n");
    printf("  - Create fresh networks for each hyperparameter combination\n");
    printf("  - Test learning rates: 0.01, 0.1, 0.5, 1.0\n");
    printf("  - Test epochs: 10000, 25000, 50000, 100000\n\n");
}

int main(int argc, char *argv[]) {
    // Parse command line arguments
    int load_mode = 0;
    char *model_path = NULL;
    Net *base_net = NULL;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_help(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "--load") == 0) {
            if (i + 1 < argc) {
                load_mode = 1;
                model_path = argv[i + 1];
                i++;
            } else {
                fprintf(stderr, "Error: --load requires a model file path\n");
                print_help(argv[0]);
                return 1;
            }
        } else {
            fprintf(stderr, "Error: Unknown option '%s'\n", argv[i]);
            print_help(argv[0]);
            return 1;
        }
    }

    double learning_rates[] = {0.01, 0.1, 0.5, 1.0};
    int rounds[] = {10000, 25000, 50000, 100000};
    int num_learning_rates = sizeof(learning_rates) / sizeof(double);
    int num_rounds = sizeof(rounds) / sizeof(int);

    if (load_mode) {
        printf("Loading base model from %s for fine-tuning...\n", model_path);
        base_net = load_net(model_path);
        if (!base_net) {
            fprintf(stderr, "Failed to load model\n");
            return 1;
        }
        printf("Model loaded successfully\n\n");
    }

    Config* config = load_config("conf/xornet.conf");
    if(!config) {
        fprintf(stderr, "Failed to load config file.\n");
        if (base_net) free_net(base_net);
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
            Net *net = NULL;

            if (load_mode) {
                // Copy the base model by saving and loading to a temp file
                if (save_net(base_net, ".gym_temp.model") != 0) {
                    fprintf(stderr, "Failed to copy model.\n");
                    continue;
                }
                net = load_net(".gym_temp.model");
                if (!net) {
                    fprintf(stderr, "Failed to load copied model.\n");
                    continue;
                }
            } else {
                net = create_net(config);
                if (!net) {
                    fprintf(stderr, "Failed to create network.\n");
                    continue;
                }
            }

            train_nn(inputs, expected, num_samples, net, rounds[j], learning_rates[i]);
            double mse = test_nn_and_get_mse(inputs, expected, num_samples, net);

            printf("| %-13.2f | %-6d | %-8.6f |\n", learning_rates[i], rounds[j], mse);

            free_net(net);
        }
    }

    if (base_net) free_net(base_net);
    free_dataset(inputs, expected, num_samples);
    free_config(config);

    return 0;
}

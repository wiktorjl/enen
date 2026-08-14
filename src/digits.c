#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "nn.h"
#include "config.h"
#include "tools.h"

void print_help(const char *program_name) {
    printf("Usage: %s [OPTIONS]\n\n", program_name);
    printf("8x8 Digit Recognition Neural Network\n\n");
    printf("Options:\n");
    printf("  --load MODEL    Load pre-trained model from file and test\n");
    printf("  --help          Display this help message\n\n");
    printf("Default behavior (no options):\n");
    printf("  - Load configuration from conf/digits.conf\n");
    printf("  - Load dataset from datasets/digits_dataset.csv\n");
    printf("  - Train new network\n");
    printf("  - Save trained model to digits.model\n\n");
    printf("Examples:\n");
    printf("  %s                                # Train and save new model\n", program_name);
    printf("  %s --load models/digits.model     # Load and test existing model\n", program_name);
}

int main(int argc, char *argv[]) {
    srand(time(NULL));

    // Parse command line arguments
    int load_mode = 0;
    char *model_path = NULL;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_help(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "--load") == 0) {
            if (i + 1 < argc) {
                load_mode = 1;
                model_path = argv[i + 1];
                i++; // Skip next argument
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

    Net *net = NULL;
    Config *config = NULL;

    if (load_mode) {
        // Load existing model
        printf("Loading model from %s...\n", model_path);
        net = load_net(model_path);
        if (!net) {
            fprintf(stderr, "Failed to load model\n");
            return 1;
        }
        printf("Model loaded successfully\n");
        print_net(net, 0);

        // Still need to load config for dataset path
        config = load_config("conf/digits.conf");
        if (!config) {
            fprintf(stderr, "Failed to load config\n");
            free_net(net);
            return 1;
        }
    } else {
        // Train new model
        config = load_config("conf/digits.conf");
        if (!config) {
            fprintf(stderr, "Failed to load config\n");
            return 1;
        }

        printf("Creating network...\n");
        net = create_net(config);
        if (!net) {
            fprintf(stderr, "Failed to create network\n");
            free_config(config);
            return 1;
        }
        print_net(net, 0);

        printf("\nLoading dataset...\n");
        double **inputs = NULL;
        double **expected = NULL;
        int num_samples = 0;

        load_dataset_multiclass(config->train_dataset_path, &inputs, &expected,
                                &num_samples, config->input_size, config->output_size);
        printf("Loaded %d training samples\n", num_samples);

#ifdef _OPENMP
        int num_threads = omp_get_max_threads();
        printf("Using %d OpenMP threads\n", num_threads);
#endif

        printf("\nTraining network for %d epochs...\n", config->epochs);
        train_nn_multiclass(inputs, expected, num_samples, net,
                           config->epochs, config->learning_rate);

        printf("Training complete!\n");


        // Now load the test dataset
        free(inputs);
        free(expected);
        load_dataset_multiclass(config->test_dataset_path, &inputs, &expected,
                            &num_samples, config->input_size, config->output_size);
        printf("Loaded %d test samples\n", num_samples);


        // Calculate and display MSE
        double mse = test_nn_and_get_mse_multiclass(inputs, expected, num_samples, net);
        printf("Final MSE: %.6f\n", mse);


        // Test the network
        test_nn_multiclass(inputs, expected, num_samples, net);

        // Save the model
        printf("\nSaving model to models/digits.model...\n");
        if (save_net(net, "models/digits.model") == 0) {
            printf("Model saved successfully\n");
        } else {
            fprintf(stderr, "Failed to save model\n");
        }

        free_dataset_multiclass(inputs, expected, num_samples);
    }

    // If in load mode, still test with the dataset
    if (load_mode) {
        printf("\nLoading dataset for testing...\n");
        double **inputs = NULL;
        double **expected = NULL;
        int num_samples = 0;

        load_dataset_multiclass(config->test_dataset_path, &inputs, &expected,
                                &num_samples, config->input_size, config->output_size);
        printf("Loaded %d samples\n", num_samples);

        // Calculate and display MSE
        double mse = test_nn_and_get_mse_multiclass(inputs, expected, num_samples, net);
        printf("Final MSE: %.6f\n", mse);

        // Test the network
        test_nn_multiclass(inputs, expected, num_samples, net);

        free_dataset_multiclass(inputs, expected, num_samples);
    }

    free_net(net);
    free_config(config);

    return 0;
}

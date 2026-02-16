#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include "nn.h"
#include "config.h"
#include "tools.h"

void print_bar(double value, double max_value, int width) {
    if (value < 0) value = 0;
    int bar_length = (int)((value / max_value) * width);
    if (bar_length < 0) bar_length = 0;
    if (bar_length > width) bar_length = width;

    printf("[");
    for (int i = 0; i < width; ++i) {
        if (i < bar_length) {
            printf("#");
        } else {
            printf(" ");
        }
    }
    printf("]");
}

void print_histogram(double* accuracies, int num_runs) {
    int counts[5] = {0};

    for (int i = 0; i < num_runs; i++) {
        int acc_val = (int)round(accuracies[i]);
        if (acc_val == 0) counts[0]++;
        else if (acc_val == 25) counts[1]++;
        else if (acc_val == 50) counts[2]++;
        else if (acc_val == 75) counts[3]++;
        else if (acc_val == 100) counts[4]++;
    }

    int max_count = 0;
    for (int i = 0; i < 5; i++) {
        if (counts[i] > max_count) {
            max_count = counts[i];
        }
    }

    printf("\n--- Accuracy Distribution Histogram ---\n");
    if (max_count == 0) {
        printf("No data to display.\n");
        return;
    }

    double levels[] = {0.0, 25.0, 50.0, 75.0, 100.0};
    for (int i = 0; i < 5; i++) {
        printf("%5.1f%% | ", levels[i]);
        print_bar(counts[i], max_count, 50);
        printf(" (%d runs)\n", counts[i]);
    }
}

void print_help(const char *program_name) {
    printf("Usage: %s <number_of_runs> [OPTIONS]\n", program_name);
    printf("       %s --load MODEL\n\n", program_name);
    printf("Statistical Accuracy Analysis Tool\n\n");
    printf("Options:\n");
    printf("  --load MODEL    Load pre-trained model and test accuracy once\n");
    printf("  --help          Display this help message\n\n");
    printf("Default behavior:\n");
    printf("  - Train N networks from scratch with different random initializations\n");
    printf("  - Report accuracy statistics across all runs\n\n");
    printf("Examples:\n");
    printf("  %s 100                         # Train 100 times, show statistics\n", program_name);
    printf("  %s --load models/xor.model     # Test pre-trained model once\n", program_name);
}

int main(int argc, char *argv[]) {
    // Parse command line arguments
    int load_mode = 0;
    char *model_path = NULL;
    int num_runs = 0;

    if (argc < 2) {
        fprintf(stderr, "Error: Missing arguments\n\n");
        print_help(argv[0]);
        return 1;
    }

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
        } else if (num_runs == 0) {
            num_runs = atoi(argv[i]);
            if (num_runs <= 0 && !load_mode) {
                fprintf(stderr, "Number of runs must be a positive integer.\n");
                return 1;
            }
        } else {
            fprintf(stderr, "Error: Unknown argument '%s'\n", argv[i]);
            print_help(argv[0]);
            return 1;
        }
    }

    if (load_mode && num_runs != 0) {
        fprintf(stderr, "Error: Cannot specify number of runs with --load\n");
        print_help(argv[0]);
        return 1;
    }

    if (!load_mode && num_runs == 0) {
        fprintf(stderr, "Error: Number of runs required\n");
        print_help(argv[0]);
        return 1;
    }

    Config* config = load_config("conf/xornet.conf");
    if(!config) {
        fprintf(stderr, "Failed to load config file.\n");
        return 1;
    }

    double **inputs = NULL;
    double *expected = NULL;
    int num_samples = 0;

    load_dataset(config->dataset_path, &inputs, &expected, &num_samples, config->input_size);

    if (load_mode) {
        // Load model and test once
        printf("Loading model from %s...\n", model_path);
        Net *net = load_net(model_path);
        if (!net) {
            fprintf(stderr, "Failed to load model\n");
            free_dataset(inputs, expected, num_samples);
            free_config(config);
            return 1;
        }
        printf("Model loaded successfully\n");
        print_net(net, 0);

        printf("\nTesting model on %d samples...\n", num_samples);
        int correct_predictions = 0;
        for (int i = 0; i < num_samples; i++) {
            forward_pass(inputs[i], net);
            int output_layer = net->num_layers - 1;
            int prediction = (net->activations[output_layer][0] > 0.5) ? 1 : 0;
            if (prediction == (int)expected[i]) {
                correct_predictions++;
            }
        }
        double accuracy = (double)correct_predictions / num_samples * 100.0;
        double mse = test_nn_and_get_mse(inputs, expected, num_samples, net);

        printf("\n=== Results ===\n");
        printf("Accuracy: %.1f%% (%d/%d correct)\n", accuracy, correct_predictions, num_samples);
        printf("MSE: %.6f\n", mse);

        free_net(net);
        free_dataset(inputs, expected, num_samples);
        free_config(config);
        return 0;
    }

    // Normal mode: train multiple times
    double* accuracies = malloc(num_runs * sizeof(double));
    if (!accuracies) {
        perror("Failed to allocate memory for accuracies");
        free_dataset(inputs, expected, num_samples);
        free_config(config);
        return 1;
    }

    srand(time(NULL));

    printf("--- Running %d training sessions... ---\n\n", num_runs);
    printf("--- Live Average Accuracy Plateau ---\n");

    double running_total_accuracy = 0.0;

    for (int run = 0; run < num_runs; run++) {
        Net *net = create_net(config);
        if (!net) {
            fprintf(stderr, "Failed to create network for run %d.\n", run);
            continue;
        }

        train_nn(inputs, expected, num_samples, net, config->epochs, config->learning_rate);

        int correct_predictions = 0;
        for (int i = 0; i < num_samples; i++) {
            forward_pass(inputs[i], net);
            int output_layer = net->num_layers - 1;
            int prediction = (net->activations[output_layer][0] > 0.5) ? 1 : 0;
            if (prediction == (int)expected[i]) {
                correct_predictions++;
            }
        }
        accuracies[run] = (double)correct_predictions / num_samples * 100.0;
        running_total_accuracy += accuracies[run];
        double running_avg = running_total_accuracy / (run + 1);

        free_net(net);

        printf("Run %4d/%-4d | Acc: %5.1f%% | Avg: %5.1f%% ", run + 1, num_runs, accuracies[run], running_avg);
        print_bar(running_avg, 100, 40);
        printf("\n");
    }

    double min_accuracy = 100.0;
    double max_accuracy = 0.0;

    for (int i = 0; i < num_runs; i++) {
        if (accuracies[i] < min_accuracy) min_accuracy = accuracies[i];
        if (accuracies[i] > max_accuracy) max_accuracy = accuracies[i];
    }
    double final_avg_accuracy = running_total_accuracy / num_runs;

    double sum_sq_diff = 0.0;
    for (int i = 0; i < num_runs; i++) {
        sum_sq_diff += pow(accuracies[i] - final_avg_accuracy, 2);
    }
    double std_dev = (num_runs > 1) ? sqrt(sum_sq_diff / (num_runs - 1)) : 0.0;
    double sem = (num_runs > 0) ? std_dev / sqrt(num_runs) : 0.0;

    printf("\n\n--- Statistical Summary over %d runs ---\n", num_runs);
    printf("Average Accuracy: %6.2f%%\n", final_avg_accuracy);
    printf("Standard Deviation: %6.2f%%\n", std_dev);
    printf("Standard Error:   %6.2f%%\n", sem);
    printf("Minimum Accuracy: %6.2f%%\n", min_accuracy);
    printf("Maximum Accuracy: %6.2f%%\n", max_accuracy);

    print_histogram(accuracies, num_runs);

    free(accuracies);
    free_dataset(inputs, expected, num_samples);
    free_config(config);

    return 0;
}

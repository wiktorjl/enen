#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include "nn.h"
#include "config.h"
#include "tools.h"

#define NUM_ACCURACY_BINS 10

static int index_of_max(const double *values, int count) {
    int max_index = 0;

    for (int i = 1; i < count; i++) {
        if (values[i] > values[max_index]) {
            max_index = i;
        }
    }

    return max_index;
}

static int count_correct_predictions(double **inputs, double **expected,
                                     int num_samples, Net *net) {
    int output_layer = net->num_layers - 1;
    int output_size = net->layer_sizes[output_layer];
    int correct_predictions = 0;

    for (int i = 0; i < num_samples; i++) {
        forward_pass(inputs[i], net);
        int prediction = index_of_max(net->activations[output_layer], output_size);
        int expected_class = index_of_max(expected[i], output_size);

        if (prediction == expected_class) {
            correct_predictions++;
        }
    }

    return correct_predictions;
}

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
    int counts[NUM_ACCURACY_BINS] = {0};

    for (int i = 0; i < num_runs; i++) {
        int bin = (int)(accuracies[i] / 10.0);
        if (bin < 0) bin = 0;
        if (bin >= NUM_ACCURACY_BINS) bin = NUM_ACCURACY_BINS - 1;
        counts[bin]++;
    }

    int max_count = 0;
    for (int i = 0; i < NUM_ACCURACY_BINS; i++) {
        if (counts[i] > max_count) {
            max_count = counts[i];
        }
    }

    printf("\n--- Accuracy Distribution Histogram ---\n");
    if (max_count == 0) {
        printf("No data to display.\n");
        return;
    }

    for (int i = 0; i < NUM_ACCURACY_BINS; i++) {
        int lower_bound = i * 10;
        int upper_bound = (i + 1) * 10;
        printf(i == NUM_ACCURACY_BINS - 1 ? "%3d-%3d%% | " : "%3d-<%3d%% | ",
               lower_bound, upper_bound);
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
    printf("  %s --load models/digits.model  # Test pre-trained model once\n", program_name);
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

    Config* config = load_config("conf/digits.conf");
    if(!config) {
        fprintf(stderr, "Failed to load config file.\n");
        return 1;
    }

    double **test_inputs = NULL;
    double **test_expected = NULL;
    int num_test_samples = 0;

    load_dataset_multiclass(config->test_dataset_path,
                            &test_inputs, &test_expected, &num_test_samples,
                            config->input_size, config->output_size);
    if (load_mode) {
        // Load model and test once
        printf("Loading model from %s...\n", model_path);
        Net *net = load_net(model_path);
        if (!net) {
            fprintf(stderr, "Failed to load model\n");
            free_dataset_multiclass(test_inputs, test_expected, num_test_samples);
            free_config(config);
            return 1;
        }

        int output_layer = net->num_layers - 1;
        if (net->layer_sizes[0] != config->input_size ||
            net->layer_sizes[output_layer] != config->output_size) {
            fprintf(stderr,
                    "Model shape does not match config (expected %d inputs and %d outputs)\n",
                    config->input_size, config->output_size);
            free_net(net);
            free_dataset_multiclass(test_inputs, test_expected, num_test_samples);
            free_config(config);
            return 1;
        }

        printf("Model loaded successfully\n");
        print_net(net, 0);

        printf("\nTesting model on %d samples...\n", num_test_samples);
        int correct_predictions = count_correct_predictions(
            test_inputs, test_expected, num_test_samples, net);
        double accuracy = (double)correct_predictions / num_test_samples * 100.0;
        double mse = test_nn_and_get_mse_multiclass(
            test_inputs, test_expected, num_test_samples, net);

        printf("\n=== Results ===\n");
        printf("Accuracy: %.1f%% (%d/%d correct)\n",
               accuracy, correct_predictions, num_test_samples);
        printf("MSE: %.6f\n", mse);

        free_net(net);
        free_dataset_multiclass(test_inputs, test_expected, num_test_samples);
        free_config(config);
        return 0;
    }

    // Normal mode: train multiple times
    double **train_inputs = NULL;
    double **train_expected = NULL;
    int num_train_samples = 0;

    load_dataset_multiclass(config->train_dataset_path,
                            &train_inputs, &train_expected, &num_train_samples,
                            config->input_size, config->output_size);

    double* accuracies = malloc(num_runs * sizeof(double));
    if (!accuracies) {
        perror("Failed to allocate memory for accuracies");
        free_dataset_multiclass(train_inputs, train_expected, num_train_samples);
        free_dataset_multiclass(test_inputs, test_expected, num_test_samples);
        free_config(config);
        return 1;
    }

    srand(time(NULL));

    printf("--- Running %d training sessions... ---\n\n", num_runs);
    printf("--- Live Average Accuracy Plateau ---\n");

    double running_total_accuracy = 0.0;
    int completed_runs = 0;

    for (int run = 0; run < num_runs; run++) {
        Net *net = create_net(config);
        if (!net) {
            fprintf(stderr, "Failed to create network for run %d.\n", run + 1);
            continue;
        }

        train_nn_multiclass(train_inputs, train_expected, num_train_samples,
                            net, config->epochs, config->learning_rate);

        int correct_predictions = count_correct_predictions(
            test_inputs, test_expected, num_test_samples, net);
        double accuracy = (double)correct_predictions / num_test_samples * 100.0;
        accuracies[completed_runs] = accuracy;
        completed_runs++;
        running_total_accuracy += accuracy;
        double running_avg = running_total_accuracy / completed_runs;

        free_net(net);

        printf("Run %4d/%-4d | Acc: %5.1f%% | Avg: %5.1f%% ",
               run + 1, num_runs, accuracy, running_avg);
        print_bar(running_avg, 100, 40);
        printf("\n");
    }

    if (completed_runs == 0) {
        fprintf(stderr, "No training runs completed successfully.\n");
        free(accuracies);
        free_dataset_multiclass(train_inputs, train_expected, num_train_samples);
        free_dataset_multiclass(test_inputs, test_expected, num_test_samples);
        free_config(config);
        return 1;
    }

    double min_accuracy = 100.0;
    double max_accuracy = 0.0;

    for (int i = 0; i < completed_runs; i++) {
        if (accuracies[i] < min_accuracy) min_accuracy = accuracies[i];
        if (accuracies[i] > max_accuracy) max_accuracy = accuracies[i];
    }
    double final_avg_accuracy = running_total_accuracy / completed_runs;

    double sum_sq_diff = 0.0;
    for (int i = 0; i < completed_runs; i++) {
        sum_sq_diff += pow(accuracies[i] - final_avg_accuracy, 2);
    }
    double std_dev = (completed_runs > 1)
        ? sqrt(sum_sq_diff / (completed_runs - 1))
        : 0.0;
    double sem = std_dev / sqrt(completed_runs);

    printf("\n\n--- Statistical Summary over %d completed runs ---\n",
           completed_runs);
    printf("Average Accuracy: %6.2f%%\n", final_avg_accuracy);
    printf("Standard Deviation: %6.2f%%\n", std_dev);
    printf("Standard Error:   %6.2f%%\n", sem);
    printf("Minimum Accuracy: %6.2f%%\n", min_accuracy);
    printf("Maximum Accuracy: %6.2f%%\n", max_accuracy);

    print_histogram(accuracies, completed_runs);

    free(accuracies);
    free_dataset_multiclass(train_inputs, train_expected, num_train_samples);
    free_dataset_multiclass(test_inputs, test_expected, num_test_samples);
    free_config(config);

    return 0;
}

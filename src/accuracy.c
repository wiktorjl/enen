#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "config.h"
#include "dataset.h"
#include "nn.h"

#define CONFIG_PATH "conf/digits.conf"
#define NUM_ACCURACY_BINS 10

static void print_bar(double value, double maximum, int width) {
    int length = maximum > 0.0 ? (int)(value / maximum * width) : 0;
    if (length < 0) length = 0;
    if (length > width) length = width;
    putchar('[');
    for (int index = 0; index < width; index++) {
        putchar(index < length ? '#' : ' ');
    }
    putchar(']');
}

static void print_histogram(const double *accuracies, int num_runs) {
    int counts[NUM_ACCURACY_BINS] = {0};
    int maximum_count = 0;
    for (int run = 0; run < num_runs; run++) {
        int bin = (int)(accuracies[run] / 10.0);
        if (bin < 0) bin = 0;
        if (bin >= NUM_ACCURACY_BINS) bin = NUM_ACCURACY_BINS - 1;
        counts[bin]++;
        if (counts[bin] > maximum_count) {
            maximum_count = counts[bin];
        }
    }

    printf("\nAccuracy distribution:\n");
    for (int bin = 0; bin < NUM_ACCURACY_BINS; bin++) {
        int lower = bin * 10;
        int upper = (bin + 1) * 10;
        printf(bin + 1 == NUM_ACCURACY_BINS ? "%3d-%3d%% | " : "%3d-<%3d%% | ",
               lower, upper);
        print_bar(counts[bin], maximum_count, 40);
        printf(" (%d)\n", counts[bin]);
    }
}

static int parse_positive_int(const char *text, int *value) {
    char *end = NULL;
    errno = 0;
    long parsed = strtol(text, &end, 10);
    if (end == text || *end != '\0' || errno == ERANGE || parsed <= 0 ||
        parsed > 1000000) {
        return -1;
    }
    *value = (int)parsed;
    return 0;
}

static void print_help(const char *program_name) {
    printf("Usage: %s RUNS\n", program_name);
    printf("       %s --load MODEL\n\n", program_name);
    printf("Measure held-out classification accuracy for the UCI digits model.\n");
    printf("RUNS trains independently initialized models and summarizes their\n");
    printf("accuracy distribution. --load evaluates one saved model.\n");
}

static int evaluate_saved_model(const char *model_path, const Config *config,
                                const Dataset *test) {
    Net *net = load_net(model_path);
    if (!net) {
        return -1;
    }
    if (!net_matches_dimensions(net, config->input_size, config->output_size)) {
        fprintf(stderr,
                "Model dimensions do not match the configured dataset "
                "(%d inputs, %d classes)\n",
                config->input_size, config->output_size);
        free_net(net);
        return -1;
    }
    print_net(net);
    ClassificationMetrics metrics = evaluate_classifier(
        test->inputs, test->targets, test->num_samples, net, NULL);
    printf("Accuracy: %.2f%% (%d/%d)\n",
           100.0 * metrics.correct / metrics.total, metrics.correct,
           metrics.total);
    printf("Cross-entropy: %.6f\n", metrics.cross_entropy);
    free_net(net);
    return 0;
}

int main(int argc, char **argv) {
    if (argc == 2 && (strcmp(argv[1], "--help") == 0 ||
                      strcmp(argv[1], "-h") == 0)) {
        print_help(argv[0]);
        return EXIT_SUCCESS;
    }

    const char *model_path = NULL;
    int num_runs = 0;
    if (argc == 3 && strcmp(argv[1], "--load") == 0) {
        model_path = argv[2];
    } else if (argc != 2 || parse_positive_int(argv[1], &num_runs) != 0) {
        print_help(argv[0]);
        return EXIT_FAILURE;
    }

    Config *config = load_config(CONFIG_PATH);
    if (!config) {
        return EXIT_FAILURE;
    }
    Dataset test = {0};
    if (load_dataset(config->test_dataset_path, config->input_size,
                     config->output_size, &test) != 0) {
        free_config(config);
        return EXIT_FAILURE;
    }

    if (model_path) {
        int status = evaluate_saved_model(model_path, config, &test);
        free_dataset(&test);
        free_config(config);
        return status == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    Dataset training = {0};
    if (load_dataset(config->train_dataset_path, config->input_size,
                     config->output_size, &training) != 0) {
        free_dataset(&test);
        free_config(config);
        return EXIT_FAILURE;
    }
    double *accuracies = malloc(sizeof(*accuracies) * (size_t)num_runs);
    if (!accuracies) {
        perror("Failed to allocate accuracy results");
        free_dataset(&training);
        free_dataset(&test);
        free_config(config);
        return EXIT_FAILURE;
    }

    srand((unsigned int)time(NULL));
    double accuracy_total = 0.0;
    int completed_runs = 0;
    printf("Training %d independently initialized classifiers...\n", num_runs);
    for (int run = 0; run < num_runs; run++) {
        Net *net = create_net(config);
        if (!net) {
            fprintf(stderr, "Run %d: network creation failed\n", run + 1);
            continue;
        }
        if (train_classifier(training.inputs, training.targets,
                             training.num_samples, net, config->epochs,
                             config->learning_rate) != 0) {
            fprintf(stderr, "Run %d: training failed\n", run + 1);
            free_net(net);
            continue;
        }

        ClassificationMetrics metrics = evaluate_classifier(
            test.inputs, test.targets, test.num_samples, net, NULL);
        double accuracy = 100.0 * metrics.correct / metrics.total;
        accuracies[completed_runs++] = accuracy;
        accuracy_total += accuracy;
        printf("Run %4d/%-4d | accuracy %6.2f%% | average %6.2f%% ",
               run + 1, num_runs, accuracy, accuracy_total / completed_runs);
        print_bar(accuracy_total / completed_runs, 100.0, 30);
        putchar('\n');
        free_net(net);
    }

    if (completed_runs == 0) {
        fprintf(stderr, "No training runs completed successfully\n");
        free(accuracies);
        free_dataset(&training);
        free_dataset(&test);
        free_config(config);
        return EXIT_FAILURE;
    }

    double mean = accuracy_total / completed_runs;
    double minimum = accuracies[0];
    double maximum = accuracies[0];
    double squared_difference_sum = 0.0;
    for (int run = 0; run < completed_runs; run++) {
        if (accuracies[run] < minimum) minimum = accuracies[run];
        if (accuracies[run] > maximum) maximum = accuracies[run];
        squared_difference_sum += pow(accuracies[run] - mean, 2.0);
    }
    double standard_deviation = completed_runs > 1
        ? sqrt(squared_difference_sum / (completed_runs - 1))
        : 0.0;
    double standard_error = standard_deviation / sqrt((double)completed_runs);

    printf("\nSummary over %d completed runs:\n", completed_runs);
    printf("Mean accuracy:       %6.2f%%\n", mean);
    printf("Standard deviation:  %6.2f%%\n", standard_deviation);
    printf("Standard error:      %6.2f%%\n", standard_error);
    printf("Minimum accuracy:    %6.2f%%\n", minimum);
    printf("Maximum accuracy:    %6.2f%%\n", maximum);
    print_histogram(accuracies, completed_runs);

    free(accuracies);
    free_dataset(&training);
    free_dataset(&test);
    free_config(config);
    return EXIT_SUCCESS;
}

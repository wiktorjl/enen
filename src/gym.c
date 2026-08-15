#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "config.h"
#include "dataset.h"
#include "nn.h"

#define CONFIG_PATH "conf/digits.conf"
#define VALIDATION_INTERVAL 5

typedef struct {
    /* Views borrow row storage from the owning Dataset. */
    double **inputs;
    double **targets;
    int count;
} DatasetView;

static void print_help(const char *program_name) {
    printf("Usage: %s [--load MODEL]\n\n", program_name);
    printf("Compare learning rates and epoch counts for UCI digit classification.\n");
    printf("One fifth of the configured training split is reserved for validation;\n");
    printf("the configured test split is never used for parameter selection.\n\n");
    printf("  --load MODEL  Fine-tune clones of a saved starting model\n");
    printf("  --help        Display this help message\n");
}

static int create_training_validation_views(const Dataset *dataset,
                                            DatasetView *training,
                                            DatasetView *validation) {
    training->count = dataset->num_samples -
                      (dataset->num_samples + VALIDATION_INTERVAL - 1) /
                          VALIDATION_INTERVAL;
    validation->count = dataset->num_samples - training->count;
    training->inputs = malloc(sizeof(*training->inputs) *
                              (size_t)training->count);
    training->targets = malloc(sizeof(*training->targets) *
                               (size_t)training->count);
    validation->inputs = malloc(sizeof(*validation->inputs) *
                                (size_t)validation->count);
    validation->targets = malloc(sizeof(*validation->targets) *
                                 (size_t)validation->count);
    if (!training->inputs || !training->targets || !validation->inputs ||
        !validation->targets) {
        perror("Failed to allocate training/validation views");
        return -1;
    }

    int training_index = 0;
    int validation_index = 0;
    for (int sample = 0; sample < dataset->num_samples; sample++) {
        DatasetView *view = sample % VALIDATION_INTERVAL == 0
            ? validation
            : training;
        int *index = sample % VALIDATION_INTERVAL == 0
            ? &validation_index
            : &training_index;
        view->inputs[*index] = dataset->inputs[sample];
        view->targets[*index] = dataset->targets[sample];
        (*index)++;
    }
    return 0;
}

static void free_view(DatasetView *view) {
    free(view->inputs);
    free(view->targets);
    view->inputs = NULL;
    view->targets = NULL;
    view->count = 0;
}

int main(int argc, char **argv) {
    const char *model_path = NULL;
    for (int argument = 1; argument < argc; argument++) {
        if (strcmp(argv[argument], "--help") == 0 ||
            strcmp(argv[argument], "-h") == 0) {
            print_help(argv[0]);
            return EXIT_SUCCESS;
        }
        if (strcmp(argv[argument], "--load") == 0 && argument + 1 < argc &&
            !model_path) {
            model_path = argv[++argument];
            continue;
        }
        fprintf(stderr, "Unknown or incomplete option: %s\n", argv[argument]);
        print_help(argv[0]);
        return EXIT_FAILURE;
    }

    Config *config = load_config(CONFIG_PATH);
    if (!config) {
        return EXIT_FAILURE;
    }
    Dataset dataset = {0};
    if (load_dataset(config->train_dataset_path, config->input_size,
                     config->output_size, &dataset) != 0) {
        free_config(config);
        return EXIT_FAILURE;
    }

    DatasetView training = {0};
    DatasetView validation = {0};
    if (create_training_validation_views(&dataset, &training, &validation) != 0) {
        free_view(&training);
        free_view(&validation);
        free_dataset(&dataset);
        free_config(config);
        return EXIT_FAILURE;
    }

    unsigned int experiment_seed = (unsigned int)time(NULL);
    srand(experiment_seed);
    Net *base_net = model_path ? load_net(model_path) : create_net(config);
    if (!base_net || !net_matches_dimensions(
            base_net, config->input_size, config->output_size)) {
        if (base_net) {
            fprintf(stderr, "Model dimensions do not match the dataset\n");
        }
        free_net(base_net);
        free_view(&training);
        free_view(&validation);
        free_dataset(&dataset);
        free_config(config);
        return EXIT_FAILURE;
    }

    const double learning_rates[] = {0.01, 0.05, 0.1, 0.5};
    const int epochs[] = {10, 25, 50, 100};
    const int rate_count = (int)(sizeof(learning_rates) /
                                 sizeof(learning_rates[0]));
    const int epoch_count = (int)(sizeof(epochs) / sizeof(epochs[0]));
    double best_accuracy = -1.0;
    double best_loss = INFINITY;
    double best_rate = 0.0;
    int best_epochs = 0;

    printf("Fitting %d samples; validating on %d samples.\n",
           training.count, validation.count);
    printf("| Learning rate | Epochs | Validation accuracy | Cross-entropy |\n");
    printf("|---------------|--------|---------------------|---------------|\n");
    for (int rate = 0; rate < rate_count; rate++) {
        for (int epoch = 0; epoch < epoch_count; epoch++) {
            Net *net = clone_net(base_net);
            if (!net) {
                fprintf(stderr, "Failed to clone the starting network\n");
                continue;
            }

            /* Identical shuffle streams make comparisons reproducible and fair. */
            srand(experiment_seed + 1U);
            if (train_classifier(training.inputs, training.targets,
                                 training.count, net, epochs[epoch],
                                 learning_rates[rate]) != 0) {
                fprintf(stderr, "Training failed for one parameter combination\n");
                free_net(net);
                continue;
            }
            ClassificationMetrics metrics = evaluate_classifier(
                validation.inputs, validation.targets, validation.count, net,
                NULL);
            double accuracy = 100.0 * metrics.correct / metrics.total;
            printf("| %13.2f | %6d | %18.2f%% | %13.6f |\n",
                   learning_rates[rate], epochs[epoch], accuracy,
                   metrics.cross_entropy);

            if (accuracy > best_accuracy ||
                (accuracy == best_accuracy && metrics.cross_entropy < best_loss)) {
                best_accuracy = accuracy;
                best_loss = metrics.cross_entropy;
                best_rate = learning_rates[rate];
                best_epochs = epochs[epoch];
            }
            free_net(net);
        }
    }

    printf("\nBest validation result: %.2f%% accuracy, %.6f cross-entropy\n",
           best_accuracy, best_loss);
    printf("Learning rate: %.2f\nEpochs: %d\n", best_rate, best_epochs);

    free_net(base_net);
    free_view(&training);
    free_view(&validation);
    free_dataset(&dataset);
    free_config(config);
    return best_accuracy >= 0.0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

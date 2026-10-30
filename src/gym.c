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
    printf("Usage: %s [--validation CSV] [--load MODEL]\n\n", program_name);
    printf("Compare learning rates and epoch counts for UCI digit classification.\n");
    printf("One fifth of the configured training split is reserved for validation;\n");
    printf("the configured test split is never used for parameter selection.\n\n");
    printf("  --validation CSV  Use a separate normalized validation dataset\n");
    printf("  --load MODEL      Fine-tune clones; requires --validation\n");
    printf("  --help            Display this help message\n\n");
    printf("Explicit validation rows must be disjoint from training and test data,\n");
    printf("and must never have been used to train the starting model.\n");
}

static int copy_dataset_view(const Dataset *dataset, DatasetView *view) {
    view->count = dataset->num_samples;
    size_t bytes = sizeof(*view->inputs) * (size_t)view->count;
    view->inputs = malloc(bytes);
    view->targets = malloc(bytes);
    if (!view->inputs || !view->targets) {
        perror("Failed to allocate dataset view");
        return -1;
    }
    memcpy(view->inputs, dataset->inputs, bytes);
    memcpy(view->targets, dataset->targets, bytes);
    return 0;
}

static int create_training_validation_views(const Dataset *dataset,
                                            DatasetView *training,
                                            DatasetView *validation) {
    if (dataset->num_samples < 2) {
        fprintf(stderr, "Tuning needs at least two samples to make a holdout\n");
        return -1;
    }
    int validation_count = 1 + (dataset->num_samples - 1) / VALIDATION_INTERVAL;
    training->count = dataset->num_samples - validation_count;
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

/* File names cannot establish independence: another CSV can contain exactly
 * the same examples. Compare feature values, even if the labels differ. */
static int datasets_overlap(const Dataset *left, const Dataset *right) {
    for (int first = 0; first < left->num_samples; first++) {
        for (int second = 0; second < right->num_samples; second++) {
            int feature = 0;
            while (feature < left->input_size &&
                   left->inputs[first][feature] == right->inputs[second][feature]) {
                feature++;
            }
            if (feature == left->input_size) {
                return 1;
            }
        }
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
    const char *validation_path = NULL;
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
        if (strcmp(argv[argument], "--validation") == 0 &&
            argument + 1 < argc && !validation_path) {
            validation_path = argv[++argument];
            continue;
        }
        fprintf(stderr, "Unknown or incomplete option: %s\n", argv[argument]);
        print_help(argv[0]);
        return EXIT_FAILURE;
    }
    if (model_path && !validation_path) {
        fprintf(stderr,
                "--load requires --validation CSV: the starting model may "
                "already have learned the default validation rows.\n");
        return EXIT_FAILURE;
    }

    Config *config = load_config(CONFIG_PATH);
    if (!config) {
        return EXIT_FAILURE;
    }
    Dataset dataset = {0};
    Dataset held_out = {0};
    Dataset test = {0};
    DatasetView training = {0};
    DatasetView validation = {0};
    Net *base_net = NULL;
    int status = EXIT_FAILURE;
    if (load_dataset(config->train_dataset_path, config->input_size,
                     config->output_size, &dataset) != 0) {
        goto cleanup;
    }

    if (validation_path) {
        if (load_dataset(validation_path, config->input_size, config->output_size,
                         &held_out) != 0 ||
            load_dataset(config->test_dataset_path, config->input_size,
                         config->output_size, &test) != 0) {
            goto cleanup;
        }
        if (datasets_overlap(&held_out, &dataset) ||
            datasets_overlap(&held_out, &test)) {
            fprintf(stderr,
                    "Validation examples overlap the configured training or "
                    "test data; supply an independent validation dataset.\n");
            goto cleanup;
        }
        if (copy_dataset_view(&dataset, &training) != 0 ||
            copy_dataset_view(&held_out, &validation) != 0) {
            goto cleanup;
        }
    } else if (create_training_validation_views(&dataset, &training,
                                               &validation) != 0) {
        goto cleanup;
    }

    unsigned int experiment_seed = (unsigned int)time(NULL);
    srand(experiment_seed);
    base_net = model_path ? load_net(model_path) : create_net(config);
    if (!base_net || !net_matches_dimensions(
            base_net, config->input_size, config->output_size)) {
        if (base_net) {
            fprintf(stderr, "Model dimensions do not match the dataset\n");
        }
        goto cleanup;
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
    status = best_accuracy >= 0.0 ? EXIT_SUCCESS : EXIT_FAILURE;

cleanup:
    free_net(base_net);
    free_view(&training);
    free_view(&validation);
    free_dataset(&dataset);
    free_dataset(&held_out);
    free_dataset(&test);
    free_config(config);
    return status;
}

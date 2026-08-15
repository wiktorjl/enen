#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "config.h"
#include "dataset.h"
#include "nn.h"

#define CONFIG_PATH "conf/digits.conf"
#define MODEL_PATH "models/digits.model"

static void print_help(const char *program_name) {
    printf("Usage: %s [--load MODEL]\n\n", program_name);
    printf("Train and evaluate the UCI handwritten-digit classifier.\n\n");
    printf("Options:\n");
    printf("  --load MODEL  Skip training and evaluate a saved model\n");
    printf("  --help        Display this help message\n\n");
    printf("Without --load, the program trains on the configured training split,\n");
    printf("evaluates on the test split, and writes %s.\n", MODEL_PATH);
}

static void report_results(const Dataset *test_data, Net *net) {
    int num_classes = test_data->num_classes;
    int *confusion = calloc((size_t)num_classes * (size_t)num_classes,
                            sizeof(*confusion));
    ClassificationMetrics metrics = evaluate_classifier(
        test_data->inputs, test_data->targets, test_data->num_samples, net,
        confusion);

    printf("\nTest accuracy: %.2f%% (%d/%d)\n",
           100.0 * metrics.correct / metrics.total, metrics.correct,
           metrics.total);
    printf("Test cross-entropy: %.6f\n", metrics.cross_entropy);
    if (confusion) {
        print_confusion_matrix(confusion, num_classes);
    }
    free(confusion);
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

    srand((unsigned int)time(NULL));
    Net *net = model_path ? load_net(model_path) : create_net(config);
    if (!net) {
        free_config(config);
        return EXIT_FAILURE;
    }
    if (!net_matches_dimensions(net, config->input_size, config->output_size)) {
        fprintf(stderr,
                "Model dimensions do not match the configured dataset "
                "(%d inputs, %d classes)\n",
                config->input_size, config->output_size);
        free_net(net);
        free_config(config);
        return EXIT_FAILURE;
    }
    print_net(net);

    Dataset training = {0};
    if (!model_path) {
        if (load_dataset(config->train_dataset_path, config->input_size,
                         config->output_size, &training) != 0) {
            free_net(net);
            free_config(config);
            return EXIT_FAILURE;
        }
        printf("Training on %d samples for %d epochs...\n",
               training.num_samples, config->epochs);
        if (train_classifier(training.inputs, training.targets,
                             training.num_samples, net, config->epochs,
                             config->learning_rate) != 0) {
            fprintf(stderr, "Training failed\n");
            free_dataset(&training);
            free_net(net);
            free_config(config);
            return EXIT_FAILURE;
        }
        free_dataset(&training);
    }

    Dataset test = {0};
    if (load_dataset(config->test_dataset_path, config->input_size,
                     config->output_size, &test) != 0) {
        free_net(net);
        free_config(config);
        return EXIT_FAILURE;
    }
    printf("Evaluating on %d held-out samples...\n", test.num_samples);
    report_results(&test, net);

    int status = EXIT_SUCCESS;
    if (!model_path) {
        if (save_net(net, MODEL_PATH) != 0) {
            status = EXIT_FAILURE;
        } else {
            printf("\nSaved model to %s\n", MODEL_PATH);
        }
    }

    free_dataset(&test);
    free_net(net);
    free_config(config);
    return status;
}

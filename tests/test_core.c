#define _POSIX_C_SOURCE 200809L

#include <limits.h>
#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

#include "config.h"
#include "dataset.h"
#include "nn.h"

#define CHECK(condition, message) do { \
    if (!(condition)) { \
        fprintf(stderr, "FAIL: %s\n", message); \
        return EXIT_FAILURE; \
    } \
} while (0)

static int write_text_file(const char *path, const char *contents) {
    FILE *file = fopen(path, "w");
    if (!file) return -1;
    int status = fputs(contents, file) == EOF ? -1 : 0;
    if (fclose(file) != 0) status = -1;
    return status;
}

static int check_rejected_inputs(void) {
    Dataset dataset = {0};
    CHECK(write_text_file("build/invalid_dataset.csv", "0.0,nan,1\n") == 0,
          "invalid dataset fixture is created");
    CHECK(load_dataset("build/invalid_dataset.csv", 2, 3, &dataset) != 0,
          "nonfinite dataset values are rejected");
    CHECK(dataset.inputs == NULL && dataset.targets == NULL,
          "rejected dataset is cleaned up");
    remove("build/invalid_dataset.csv");

    CHECK(write_text_file("build/invalid.conf",
                          "input_size=64\ninput_size=64\n") == 0,
          "invalid config fixture is created");
    Config *config = load_config("build/invalid.conf");
    CHECK(config == NULL, "duplicate config keys are rejected");
    remove("build/invalid.conf");

    CHECK(write_text_file("build/truncated.model", "ENEN") == 0,
          "truncated model fixture is created");
    Net *net = load_net("build/truncated.model");
    CHECK(net == NULL, "truncated models are rejected");
    remove("build/truncated.model");
    return EXIT_SUCCESS;
}

static int check_dataset(const Dataset *dataset, int expected_samples) {
    int class_counts[10] = {0};
    CHECK(dataset->num_samples == expected_samples,
          "dataset has the expected sample count");
    CHECK(dataset->input_size == 64, "dataset has 64 input features");
    CHECK(dataset->num_classes == 10, "dataset has 10 classes");

    for (int sample = 0; sample < dataset->num_samples; sample++) {
        double target_sum = 0.0;
        int label = -1;
        for (int feature = 0; feature < dataset->input_size; feature++) {
            CHECK(dataset->inputs[sample][feature] >= 0.0 &&
                  dataset->inputs[sample][feature] <= 1.0,
                  "features are normalized");
        }
        for (int class_id = 0; class_id < dataset->num_classes; class_id++) {
            double target = dataset->targets[sample][class_id];
            CHECK(target == 0.0 || target == 1.0, "targets are one-hot values");
            if (target == 1.0) label = class_id;
            target_sum += target;
        }
        CHECK(target_sum == 1.0, "each target has exactly one class");
        class_counts[label]++;
    }
    for (int class_id = 0; class_id < dataset->num_classes; class_id++) {
        CHECK(class_counts[class_id] > 0, "every digit class is represented");
    }
    return EXIT_SUCCESS;
}

static double example_loss(Net *net, const double *input, int label) {
    forward_pass(input, net);
    return -log(net->activations[net->num_layers - 1][label]);
}

/* A numerical derivative is independent of backpropagation's implementation.
 * Check every weight and bias, including two hidden layers so that using an
 * already-updated weight during the backward pass would fail this test. */
static int check_gradients(void) {
    int hidden_sizes[] = {4, 2};
    Config config = {
        .input_size = 3,
        .output_size = 3,
        .num_hidden_layers = 2,
        .hidden_layer_sizes = hidden_sizes,
    };
    const double input[] = {0.15, 0.35, 0.55};
    const double target[] = {0.0, 1.0, 0.0};
    srand(17);
    Net *net = create_net(&config);
    CHECK(net != NULL, "gradient-check network is created");
    Net *updated = clone_net(net);
    CHECK(updated != NULL, "gradient-check network is cloned");
    forward_pass(input, updated);
    backward_pass(target, updated, 1.0);

    const double step = 1e-5;
    for (int layer = 0; layer < net->num_layers - 1; layer++) {
        size_t weight_count = (size_t)net->layer_sizes[layer] *
                              (size_t)net->layer_sizes[layer + 1];
        for (int kind = 0; kind < 2; kind++) {
            double *parameters = kind == 0 ? net->weights[layer] : net->biases[layer];
            const double *after = kind == 0
                ? updated->weights[layer]
                : updated->biases[layer];
            size_t count = kind == 0
                ? weight_count
                : (size_t)net->layer_sizes[layer + 1];
            for (size_t index = 0; index < count; index++) {
                double original = parameters[index];
                parameters[index] = original + step;
                double plus = example_loss(net, input, 1);
                parameters[index] = original - step;
                double minus = example_loss(net, input, 1);
                parameters[index] = original;

                double numerical_gradient = (plus - minus) / (2.0 * step);
                double backward_gradient = original - after[index];
                CHECK(fabs(numerical_gradient - backward_gradient) < 1e-7,
                      "backpropagation matches the numerical loss derivative");
            }
        }
    }
    free_net(updated);
    free_net(net);

    config.num_hidden_layers = INT_MAX;
    CHECK(create_net(&config) == NULL,
          "invalid layer counts are rejected before addition or allocation");
    return EXIT_SUCCESS;
}

static int check_failed_model_save(Net *net, const char *path) {
    /* Restrict only the child: this deterministically simulates a late write
     * failure without filling a disk or changing the parent's resource limits. */
    pid_t child = fork();
    CHECK(child >= 0, "checkpoint failure child is created");
    if (child == 0) {
        struct rlimit limit;
        if (signal(SIGXFSZ, SIG_IGN) == SIG_ERR ||
            getrlimit(RLIMIT_FSIZE, &limit) != 0) {
            _exit(2);
        }
        limit.rlim_cur = 32;
        if (setrlimit(RLIMIT_FSIZE, &limit) != 0) {
            _exit(2);
        }
        _exit(save_net(net, path) == -1 ? 0 : 1);
    }
    int status;
    CHECK(waitpid(child, &status, 0) == child && WIFEXITED(status) &&
              WEXITSTATUS(status) == 0,
          "the simulated checkpoint write fails cleanly");
    Net *preserved = load_net(path);
    CHECK(preserved != NULL, "the previous checkpoint survives a failed save");
    for (int layer = 0; layer < net->num_layers - 1; layer++) {
        size_t weights_bytes = sizeof(double) * (size_t)net->layer_sizes[layer] *
                               (size_t)net->layer_sizes[layer + 1];
        size_t biases_bytes = sizeof(double) *
                              (size_t)net->layer_sizes[layer + 1];
        CHECK(memcmp(net->weights[layer], preserved->weights[layer],
                     weights_bytes) == 0 &&
                  memcmp(net->biases[layer], preserved->biases[layer],
                         biases_bytes) == 0,
              "the previous checkpoint retains every parameter exactly");
    }
    free_net(preserved);
    return EXIT_SUCCESS;
}

static int check_network(void) {
    int hidden_size = 6;
    Config config = {
        .input_size = 2,
        .output_size = 3,
        .num_hidden_layers = 1,
        .hidden_layer_sizes = &hidden_size,
        .learning_rate = 0.1,
        .epochs = 500
    };
    double input_rows[3][2] = {{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}};
    double target_rows[3][3] = {{1.0, 0.0, 0.0},
                                {0.0, 1.0, 0.0},
                                {0.0, 0.0, 1.0}};
    double *inputs[3] = {input_rows[0], input_rows[1], input_rows[2]};
    double *targets[3] = {target_rows[0], target_rows[1], target_rows[2]};

    srand(7);
    Net *net = create_net(&config);
    CHECK(net != NULL, "network creation succeeds");
    forward_pass(inputs[0], net);
    double probability_sum = 0.0;
    for (int class_id = 0; class_id < 3; class_id++) {
        CHECK(net->activations[2][class_id] > 0.0 &&
              net->activations[2][class_id] < 1.0,
              "softmax outputs are probabilities");
        probability_sum += net->activations[2][class_id];
    }
    CHECK(fabs(probability_sum - 1.0) < 1e-12,
          "softmax probabilities sum to one");

    ClassificationMetrics before = evaluate_classifier(inputs, targets, 3,
                                                         net, NULL);
    CHECK(train_classifier(inputs, targets, 3, net, config.epochs,
                           config.learning_rate) == 0,
          "training succeeds");
    ClassificationMetrics after = evaluate_classifier(inputs, targets, 3,
                                                        net, NULL);
    CHECK(after.cross_entropy < before.cross_entropy,
          "training lowers cross-entropy");
    CHECK(after.correct == 3, "trained network classifies the toy set");

    CHECK(save_net(net, "build/test.model") == 0, "model save succeeds");
    CHECK(check_failed_model_save(net, "build/test.model") == EXIT_SUCCESS,
          "checkpoint replacement preserves the previous model on failure");
    Net *loaded = load_net("build/test.model");
    CHECK(loaded != NULL, "model load succeeds");
    CHECK(net_matches_dimensions(loaded, 2, 3),
          "loaded model retains its dimensions");
    for (int sample = 0; sample < 3; sample++) {
        CHECK(predict_class(inputs[sample], net) ==
              predict_class(inputs[sample], loaded),
              "loaded model retains its predictions");
    }

    Net *clone = clone_net(net);
    CHECK(clone != NULL, "network clone succeeds");
    CHECK(predict_class(inputs[1], clone) == predict_class(inputs[1], net),
          "network clone retains predictions");

    remove("build/test.model");
    free_net(clone);
    free_net(loaded);
    free_net(net);
    return EXIT_SUCCESS;
}

int main(void) {
    Config *config = load_config("conf/digits.conf");
    CHECK(config != NULL, "project configuration loads");
    CHECK(config->input_size == 64, "configuration uses 64 pixels");
    CHECK(config->output_size == 10, "configuration uses ten digit classes");

    Dataset training = {0};
    Dataset test = {0};
    CHECK(load_dataset(config->train_dataset_path, config->input_size,
                       config->output_size, &training) == 0,
          "training dataset loads");
    CHECK(load_dataset(config->test_dataset_path, config->input_size,
                       config->output_size, &test) == 0,
          "test dataset loads");
    CHECK(check_dataset(&training, 3823) == EXIT_SUCCESS,
          "training dataset is valid");
    CHECK(check_dataset(&test, 1797) == EXIT_SUCCESS,
          "test dataset is valid");
    CHECK(load_optdigits_dataset("datasets/optdigits.tes", &test) == 0,
          "loading into an initialized dataset safely replaces its contents");
    CHECK(check_dataset(&test, 1797) == EXIT_SUCCESS,
          "replacement dataset remains valid");
    CHECK(check_network() == EXIT_SUCCESS, "network behavior is valid");
    CHECK(check_gradients() == EXIT_SUCCESS, "backpropagation gradients are valid");
    CHECK(check_rejected_inputs() == EXIT_SUCCESS,
          "malformed inputs are rejected safely");

    free_dataset(&training);
    free_dataset(&test);
    free_config(config);
    printf("All core tests passed.\n");
    return EXIT_SUCCESS;
}

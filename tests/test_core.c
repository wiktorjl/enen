#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    CHECK(check_network() == EXIT_SUCCESS, "network behavior is valid");
    CHECK(check_rejected_inputs() == EXIT_SUCCESS,
          "malformed inputs are rejected safely");

    free_dataset(&training);
    free_dataset(&test);
    free_config(config);
    printf("All core tests passed.\n");
    return EXIT_SUCCESS;
}

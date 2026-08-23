#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "nn.h"
#include "web_api.h"

#define CHECK(condition, message) do { \
    if (!(condition)) { \
        fprintf(stderr, "FAIL: %s\n", message); \
        web_cleanup(); \
        return EXIT_FAILURE; \
    } \
} while (0)

static int write_model_header(const char *path,
                              const int32_t layer_sizes[4]) {
    FILE *file = fopen(path, "wb");
    if (!file) {
        return -1;
    }
    const uint32_t prefix[] = {UINT32_C(0x4E454E45), UINT32_C(1), UINT32_C(4)};
    int status = fwrite(prefix, sizeof(*prefix), 3, file) == 3 &&
                         fwrite(layer_sizes, sizeof(*layer_sizes), 4, file) == 4
                     ? 0
                     : -1;
    if (fclose(file) != 0) {
        status = -1;
    }
    return status;
}

static int copy_file_with_trailing_byte(const char *source_path,
                                        const char *destination_path) {
    FILE *source = fopen(source_path, "rb");
    if (!source) {
        return -1;
    }
    FILE *destination = fopen(destination_path, "wb");
    if (!destination) {
        fclose(source);
        return -1;
    }

    unsigned char buffer[4096];
    int status = 0;
    size_t count;
    while ((count = fread(buffer, 1, sizeof(buffer), source)) > 0) {
        if (fwrite(buffer, 1, count, destination) != count) {
            status = -1;
            break;
        }
    }
    if (ferror(source) || (status == 0 && fputc(0, destination) == EOF)) {
        status = -1;
    }
    if (fclose(source) != 0) {
        status = -1;
    }
    if (fclose(destination) != 0) {
        status = -1;
    }
    return status;
}

int main(void) {
    const char *saved_model_path = "build/web_api_roundtrip.model";
    const char *malformed_model_path = "build/web_api_malformed.model";
    const char *wrong_shape_model_path = "build/web_api_wrong_shape.model";
    const char *oversized_header_model_path =
        "build/web_api_oversized_header.model";
    const char *truncated_model_path = "build/web_api_truncated.model";
    const char *trailing_model_path = "build/web_api_trailing.model";
    remove(saved_model_path);
    remove(malformed_model_path);
    remove(wrong_shape_model_path);
    remove(oversized_header_model_path);
    remove(truncated_model_path);
    remove(trailing_model_path);

    CHECK(web_initialize("datasets/optdigits.tra", "datasets/optdigits.tes") == 0,
          "browser API loads and normalizes the original UCI files");
    CHECK(web_training_samples() == 3823,
          "browser API exposes every training sample");
    CHECK(web_test_samples() == 1797,
          "browser API exposes every held-out sample");

    double *input = web_input_buffer();
    for (int pixel = 0; pixel < 64; pixel++) {
        input[pixel] = pixel % 9 == 0 ? 0.5 : 0.0;
    }
    CHECK(web_add_synthetic_sample(3) == 0,
          "browser API copies a normalized handwriting sample");
    CHECK(web_synthetic_samples() == 1,
          "browser API counts added handwriting samples");
    CHECK(web_clear_synthetic_samples() == 0 && web_synthetic_samples() == 0,
          "browser API can replace its generated handwriting set");
    for (int label = 0; label < 10; label++) {
        for (int pixel = 0; pixel < 64; pixel++) {
            input[pixel] = pixel == 8 + label * 5 ? 0.8 : 0.0;
        }
        CHECK(web_add_synthetic_sample(label) == 0,
              "browser API retains a batch of handwriting samples");
    }
    CHECK(web_synthetic_samples() == 10,
          "combined C training includes retained handwriting samples");
    CHECK(web_configure_model(3, 16, 5) != 0,
          "browser API rejects hidden layers that are too small");
    CHECK(web_configure_model(24, 16, 11) == 0,
          "browser API creates a user-selected architecture");
    CHECK(web_configure_model(257, 16, 5) != 0 && web_layer_size(1) == 24,
          "invalid configuration leaves the current model intact");
    CHECK(web_num_layers() == 4 && web_layer_size(0) == 64 &&
              web_layer_size(1) == 24 && web_layer_size(2) == 16 &&
              web_layer_size(3) == 10,
          "browser API exposes the configured layer shape");
    CHECK(web_copy_training_sample(3823) == 0,
          "browser API copies a generated training example for visualization");
    CHECK(web_inspect_input() >= 0,
          "browser API runs a visualization-only forward pass");
    for (int layer = 0; layer < web_num_layers(); layer++) {
        for (int node = 0; node < web_layer_size(layer); node++) {
            double activation = web_activation(layer, node);
            CHECK(isfinite(activation) && activation >= 0.0 && activation <= 1.0,
                  "visualized node activations are finite and normalized");
        }
    }
    CHECK(web_activation_count() == 64 + 24 + 16 + 10,
          "browser API allocates one flattened activation snapshot");
    unsigned int first_version = web_activation_version();
    CHECK(web_train_batch(7, 0.05) == 7 && web_epoch_position() == 7,
          "browser API advances a bounded training batch");
    CHECK(web_save_model(saved_model_path) != 0,
          "browser API refuses to save during a partial shuffled epoch");
    CHECK(web_activation_version() == first_version + 1 &&
              web_last_training_label() >= 0 &&
              web_last_training_label() < 10,
          "a real training batch publishes its sample and activations");
    CHECK(web_clear_synthetic_samples() != 0 &&
              web_add_synthetic_sample(3) != 0,
          "training data cannot change during a partial shuffled epoch");
    const double *snapshot = web_activation_snapshot();
    CHECK(snapshot != NULL, "browser API exposes the activation snapshot");
    double output_sum = 0.0;
    for (int node = 0; node < web_activation_count(); node++) {
        CHECK(isfinite(snapshot[node]) && snapshot[node] >= 0.0 &&
                  snapshot[node] <= 1.0,
              "training snapshot values are finite and normalized");
    }
    for (int node = web_activation_count() - 10;
         node < web_activation_count(); node++) {
        output_sum += snapshot[node];
    }
    CHECK(fabs(output_sum - 1.0) < 1e-9,
          "training snapshot output activations form a distribution");
    while (web_epochs_trained() == 0) {
        CHECK(web_train_batch(137, 0.05) > 0,
              "varied batch sizes finish exactly one epoch");
    }
    CHECK(web_epochs_trained() == 1 && web_epoch_position() == 0,
          "batch training records one epoch boundary exactly once");
    double batched_outputs[80];
    for (int sample = 0; sample < 8; sample++) {
        CHECK(web_copy_test_sample(sample) >= 0 && web_inspect_input() >= 0,
              "batched model can inspect a held-out sample");
        for (int digit = 0; digit < 10; digit++) {
            batched_outputs[sample * 10 + digit] =
                web_activation(web_num_layers() - 1, digit);
        }
    }

    CHECK(web_save_model(saved_model_path) == 0,
          "browser API saves a valid model at an epoch boundary");
    CHECK(copy_file_with_trailing_byte(saved_model_path,
                                       trailing_model_path) == 0,
          "persistence test appends a byte to a valid model fixture");
    FILE *malformed_file = fopen(malformed_model_path, "wb");
    CHECK(malformed_file != NULL,
          "persistence test creates a malformed model fixture");
    CHECK(fputs("not an enen model", malformed_file) >= 0 &&
              fclose(malformed_file) == 0,
          "persistence test writes a malformed model fixture");

    const int32_t oversized_header_sizes[] = {64, 1000000, 4, 10};
    const int32_t truncated_sizes[] = {64, 24, 16, 10};
    CHECK(write_model_header(oversized_header_model_path,
                             oversized_header_sizes) == 0 &&
              write_model_header(truncated_model_path, truncated_sizes) == 0,
          "persistence test writes allocation and length attack fixtures");

    int wrong_hidden_sizes[] = {24, 16};
    Config wrong_config = {
        .input_size = 63,
        .output_size = 10,
        .num_hidden_layers = 2,
        .hidden_layer_sizes = wrong_hidden_sizes,
        .learning_rate = 0.05,
        .epochs = 1,
    };
    Net *wrong_shape_model = create_net(&wrong_config);
    CHECK(wrong_shape_model != NULL &&
              save_net(wrong_shape_model, wrong_shape_model_path) == 0,
          "persistence test creates a valid model with the wrong input shape");
    free_net(wrong_shape_model);

    int preserved_epochs = web_epochs_trained();
    unsigned int preserved_version = web_activation_version();
    CHECK(web_load_model(malformed_model_path) != 0 &&
              web_load_model(wrong_shape_model_path) != 0 &&
              web_load_model(oversized_header_model_path) != 0 &&
              web_load_model(truncated_model_path) != 0 &&
              web_load_model(trailing_model_path) != 0,
          "browser API rejects malformed, incompatible, oversized, truncated, "
          "and trailing-byte models");
    CHECK(web_num_layers() == 4 && web_layer_size(0) == 64 &&
              web_layer_size(1) == 24 && web_layer_size(2) == 16 &&
              web_layer_size(3) == 10 &&
              web_epochs_trained() == preserved_epochs &&
              web_activation_version() == preserved_version,
          "rejected model files preserve the active model and training state");
    for (int sample = 0; sample < 8; sample++) {
        CHECK(web_copy_test_sample(sample) >= 0 && web_inspect_input() >= 0,
              "preserved model can inspect a held-out sample");
        for (int digit = 0; digit < 10; digit++) {
            CHECK(web_activation(web_num_layers() - 1, digit) ==
                      batched_outputs[sample * 10 + digit],
                  "rejected loads preserve active model predictions exactly");
        }
    }

    int retained_training_samples = web_training_samples();
    int retained_test_samples = web_test_samples();
    int retained_synthetic_samples = web_synthetic_samples();
    CHECK(web_configure_model(32, 20, 99) == 0 &&
              web_train_batch(5, 0.04) == 5,
          "persistence test creates different in-progress model state");
    CHECK(web_evaluate() == 0 && web_copy_test_sample(0) >= 0 &&
              web_predict() >= 0,
          "persistence test populates metrics and probabilities before load");
    CHECK(web_load_model(saved_model_path) == 0,
          "browser API loads a compatible saved model");
    CHECK(web_num_layers() == 4 && web_layer_size(0) == 64 &&
              web_layer_size(1) == 24 && web_layer_size(2) == 16 &&
              web_layer_size(3) == 10,
          "browser API restores the saved model architecture");
    CHECK(web_training_samples() == retained_training_samples &&
              web_test_samples() == retained_test_samples &&
              web_synthetic_samples() == retained_synthetic_samples,
          "loading a model preserves official and synthetic datasets");
    CHECK(web_epochs_trained() == 0 && web_epoch_position() == 0 &&
              web_activation_version() == 0 &&
              web_last_training_label() == -1 &&
              web_activation_snapshot() != NULL &&
              web_activation_count() == 64 + 24 + 16 + 10,
          "loading a model resets training and visualization state");
    CHECK(web_accuracy() == 0.0 && web_loss() == 0.0,
          "loading a model resets held-out metrics");
    for (int digit = 0; digit < 10; digit++) {
        CHECK(web_probability(digit) == 0.0,
              "loading a model resets prediction probabilities");
    }
    for (int pixel = 0; pixel < 64; pixel++) {
        CHECK(web_input_buffer()[pixel] == 0.0,
              "loading a model resets the shared input buffer");
    }
    for (int sample = 0; sample < 8; sample++) {
        CHECK(web_copy_test_sample(sample) >= 0 && web_inspect_input() >= 0,
              "loaded model can inspect a held-out sample");
        for (int digit = 0; digit < 10; digit++) {
            CHECK(web_activation(web_num_layers() - 1, digit) ==
                      batched_outputs[sample * 10 + digit],
                  "model persistence round-trips predictions exactly");
        }
    }

    CHECK(web_reset_model(11) == 0 && web_layer_size(1) == 24 &&
              web_layer_size(2) == 16 && web_train_epoch(0.05) == 0,
          "one-call epoch training starts from the same configured model");
    for (int sample = 0; sample < 8; sample++) {
        CHECK(web_copy_test_sample(sample) >= 0 && web_inspect_input() >= 0,
              "one-call model can inspect a held-out sample");
        for (int digit = 0; digit < 10; digit++) {
            CHECK(fabs(web_activation(web_num_layers() - 1, digit) -
                       batched_outputs[sample * 10 + digit]) < 1e-12,
                  "batch boundaries do not change training results");
        }
    }
    CHECK(web_configure_model(128, 64, 5) == 0,
          "browser API restores the deterministic reference architecture");

    for (int epoch = 0; epoch < 25; epoch++) {
        CHECK(web_train_epoch(0.05) == 0,
              "browser API trains one complete C epoch");
    }
    CHECK(web_epochs_trained() == 25, "browser API tracks completed epochs");
    CHECK(web_evaluate() == 0, "browser API evaluates the trained model");
    CHECK(web_accuracy() >= 0.95,
          "browser training reaches useful held-out accuracy");
    CHECK(web_loss() < 0.30,
          "browser training reaches useful held-out cross-entropy");

    int robust_correct = 0;
    for (int sample = 0; sample < web_test_samples(); sample++) {
        int actual = web_copy_test_sample(sample);
        CHECK(actual >= 0 && actual < 10,
              "browser API copies a valid held-out sample");
        int predicted = web_predict();
        CHECK(predicted >= 0 && predicted < 10,
              "browser API returns a valid digit prediction");
        robust_correct += predicted == actual;
    }
    double robust_accuracy = (double)robust_correct / web_test_samples();
    CHECK(robust_accuracy >= 0.94,
          "drawing-oriented robust inference preserves useful accuracy");

    CHECK(web_copy_test_sample(0) >= 0, "sample can be copied for probability test");
    CHECK(web_predict() >= 0, "probability test prediction succeeds");
    double probability_sum = 0.0;
    for (int digit = 0; digit < 10; digit++) {
        double probability = web_probability(digit);
        CHECK(isfinite(probability) && probability >= 0.0 && probability <= 1.0,
              "browser probabilities are finite and bounded");
        probability_sum += probability;
    }
    CHECK(fabs(probability_sum - 1.0) < 1e-9,
          "browser probabilities form one distribution");

    printf("Web API test accuracy: %.2f%%; robust inference: %.2f%%\n",
           web_accuracy() * 100.0, robust_accuracy * 100.0);
    web_cleanup();
    remove(saved_model_path);
    remove(malformed_model_path);
    remove(wrong_shape_model_path);
    remove(oversized_header_model_path);
    remove(truncated_model_path);
    remove(trailing_model_path);
    return EXIT_SUCCESS;
}

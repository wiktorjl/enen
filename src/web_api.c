#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#define WEB_EXPORT EMSCRIPTEN_KEEPALIVE
#else
#define WEB_EXPORT
#endif

#include "config.h"
#include "dataset.h"
#include "nn.h"
#include "web_api.h"

#define INPUT_SIZE 64
#define OUTPUT_SIZE 10
#define PREDICTION_VARIANTS 6
#define MIN_HIDDEN_SIZE 4
#define MAX_HIDDEN_SIZE 256
#define WEB_MODEL_MAGIC UINT32_C(0x4E454E45)
#define WEB_MODEL_VERSION UINT32_C(1)
#define WEB_MODEL_LAYER_COUNT 4

static Dataset training_data;
static Dataset test_data;
static Dataset synthetic_data;
static Net *model;
static double input_buffer[INPUT_SIZE];
static double prediction_probabilities[OUTPUT_SIZE];
static ClassificationMetrics latest_metrics;
static int completed_epochs;
static int configured_hidden_sizes[] = {128, 64};
static int *training_order;
static int training_order_count;
static int training_order_position;
static int training_order_ready;
static double *activation_snapshot;
static int activation_snapshot_count;
static int last_training_label = -1;
static unsigned int activation_version;

static int target_label(const double *target) {
    int label = 0;
    for (int digit = 1; digit < OUTPUT_SIZE; digit++) {
        if (target[digit] > target[label]) {
            label = digit;
        }
    }
    return label;
}

static void clear_training_progress(void) {
    free(training_order);
    training_order = NULL;
    training_order_count = 0;
    training_order_position = 0;
    training_order_ready = 0;
    free(activation_snapshot);
    activation_snapshot = NULL;
    activation_snapshot_count = 0;
    last_training_label = -1;
    activation_version = 0;
    completed_epochs = 0;
}

static int model_activation_count(const Net *candidate) {
    int count = 0;
    if (!candidate) {
        return 0;
    }
    for (int layer = 0; layer < candidate->num_layers; layer++) {
        count += candidate->layer_sizes[layer];
    }
    return count;
}

static int valid_web_model_architecture(const Net *candidate) {
    return candidate && candidate->num_layers == WEB_MODEL_LAYER_COUNT &&
           candidate->layer_sizes[0] == INPUT_SIZE &&
           candidate->layer_sizes[1] >= MIN_HIDDEN_SIZE &&
           candidate->layer_sizes[1] <= MAX_HIDDEN_SIZE &&
           candidate->layer_sizes[2] >= MIN_HIDDEN_SIZE &&
           candidate->layer_sizes[2] <= MAX_HIDDEN_SIZE &&
           candidate->layer_sizes[3] == OUTPUT_SIZE;
}

/*
 * The general nn.c loader accepts larger native networks. Browser persistence
 * is deliberately narrower, so inspect its fixed header and payload length
 * before load_net() has a chance to allocate from untrusted dimensions.
 * load_net() remains authoritative and repeats the format validation.
 */
static int add_serialized_items(size_t *total, size_t count,
                                size_t item_size) {
    if (count > (SIZE_MAX - *total) / item_size) {
        return -1;
    }
    *total += count * item_size;
    return 0;
}

static int serialized_web_model_size(const int32_t *layer_sizes,
                                     size_t *result) {
    size_t total = 0;
    if (add_serialized_items(&total, 3, sizeof(uint32_t)) != 0 ||
        add_serialized_items(&total, WEB_MODEL_LAYER_COUNT,
                             sizeof(int32_t)) != 0) {
        return -1;
    }

    for (int layer = 0; layer < WEB_MODEL_LAYER_COUNT - 1; layer++) {
        size_t current = (size_t)layer_sizes[layer];
        size_t next = (size_t)layer_sizes[layer + 1];
        if (current > SIZE_MAX / next ||
            add_serialized_items(&total, current * next,
                                 sizeof(double)) != 0) {
            return -1;
        }
    }
    for (int layer = 1; layer < WEB_MODEL_LAYER_COUNT; layer++) {
        if (add_serialized_items(&total, (size_t)layer_sizes[layer],
                                 sizeof(double)) != 0) {
            return -1;
        }
    }
    *result = total;
    return 0;
}

static int preflight_web_model_file(const char *path) {
    FILE *file = fopen(path, "rb");
    if (!file) {
        return -1;
    }

    uint32_t prefix[3];
    int32_t layer_sizes[WEB_MODEL_LAYER_COUNT];
    int status = -1;
    if (fread(prefix, sizeof(*prefix), 3, file) != 3 ||
        fread(layer_sizes, sizeof(*layer_sizes), WEB_MODEL_LAYER_COUNT, file) !=
            WEB_MODEL_LAYER_COUNT ||
        prefix[0] != WEB_MODEL_MAGIC || prefix[1] != WEB_MODEL_VERSION ||
        prefix[2] != WEB_MODEL_LAYER_COUNT ||
        layer_sizes[0] != INPUT_SIZE ||
        layer_sizes[1] < MIN_HIDDEN_SIZE ||
        layer_sizes[1] > MAX_HIDDEN_SIZE ||
        layer_sizes[2] < MIN_HIDDEN_SIZE ||
        layer_sizes[2] > MAX_HIDDEN_SIZE ||
        layer_sizes[3] != OUTPUT_SIZE) {
        goto done;
    }

    size_t expected_size = 0;
    if (serialized_web_model_size(layer_sizes, &expected_size) != 0 ||
        fseek(file, 0, SEEK_END) != 0) {
        goto done;
    }
    long actual_size = ftell(file);
    if (actual_size >= 0 && (uintmax_t)actual_size == (uintmax_t)expected_size) {
        status = 0;
    }

done:
    if (fclose(file) != 0) {
        status = -1;
    }
    return status;
}

static int replace_model(int first_hidden_size, int second_hidden_size,
                         unsigned int seed) {
    int hidden_sizes[] = {first_hidden_size, second_hidden_size};
    Config config = {
        .input_size = INPUT_SIZE,
        .output_size = OUTPUT_SIZE,
        .num_hidden_layers = 2,
        .hidden_layer_sizes = hidden_sizes,
        .learning_rate = 0.05,
        .epochs = 25,
    };

    srand(seed);
    Net *candidate = create_net(&config);
    if (!candidate) {
        return -1;
    }
    int snapshot_count = model_activation_count(candidate);
    double *candidate_snapshot = calloc((size_t)snapshot_count,
                                        sizeof(*candidate_snapshot));
    if (!candidate_snapshot) {
        free_net(candidate);
        return -1;
    }

    free_net(model);
    model = candidate;
    clear_training_progress();
    activation_snapshot = candidate_snapshot;
    activation_snapshot_count = snapshot_count;
    configured_hidden_sizes[0] = first_hidden_size;
    configured_hidden_sizes[1] = second_hidden_size;
    memset(input_buffer, 0, sizeof(input_buffer));
    memset(prediction_probabilities, 0, sizeof(prediction_probabilities));
    memset(&latest_metrics, 0, sizeof(latest_metrics));
    return 0;
}

static double *combined_training_input(int sample_index) {
    if (sample_index < training_data.num_samples) {
        return training_data.inputs[sample_index];
    }
    return synthetic_data.inputs[sample_index - training_data.num_samples];
}

static double *combined_training_target(int sample_index) {
    if (sample_index < training_data.num_samples) {
        return training_data.targets[sample_index];
    }
    return synthetic_data.targets[sample_index - training_data.num_samples];
}

static void shuffle_training_order(int *order, int count) {
    for (int index = count - 1; index > 0; index--) {
        int replacement = rand() % (index + 1);
        int temporary = order[index];
        order[index] = order[replacement];
        order[replacement] = temporary;
    }
}

static int prepare_training_order(void) {
    if (training_order_ready) {
        return 0;
    }
    int total_samples = training_data.num_samples + synthetic_data.num_samples;
    if (total_samples <= 0) {
        return -1;
    }
    if (training_order_count != total_samples) {
        int *replacement = realloc(training_order,
                                   sizeof(*replacement) *
                                       (size_t)total_samples);
        if (!replacement) {
            return -1;
        }
        training_order = replacement;
        training_order_count = total_samples;
    }
    for (int sample = 0; sample < total_samples; sample++) {
        training_order[sample] = sample;
    }
    shuffle_training_order(training_order, total_samples);
    training_order_position = 0;
    training_order_ready = 1;
    return 0;
}

static void capture_activations(void) {
    int offset = 0;
    for (int layer = 0; layer < model->num_layers; layer++) {
        int size = model->layer_sizes[layer];
        memcpy(activation_snapshot + offset, model->activations[layer],
               sizeof(*activation_snapshot) * (size_t)size);
        offset += size;
    }
    activation_version++;
}

static void translate_pixels(const double *input, double *output,
                             double offset_x, double offset_y) {
    memset(output, 0, sizeof(*output) * INPUT_SIZE);
    for (int row = 0; row < 8; row++) {
        for (int column = 0; column < 8; column++) {
            double source_x = column - offset_x;
            double source_y = row - offset_y;
            int left = (int)floor(source_x);
            int top = (int)floor(source_y);
            double fraction_x = source_x - left;
            double fraction_y = source_y - top;
            double value = 0.0;

            for (int y_step = 0; y_step <= 1; y_step++) {
                int source_row = top + y_step;
                if (source_row < 0 || source_row >= 8) {
                    continue;
                }
                double weight_y = y_step ? fraction_y : 1.0 - fraction_y;
                for (int x_step = 0; x_step <= 1; x_step++) {
                    int source_column = left + x_step;
                    if (source_column < 0 || source_column >= 8) {
                        continue;
                    }
                    double weight_x = x_step ? fraction_x : 1.0 - fraction_x;
                    value += input[source_row * 8 + source_column] *
                             weight_x * weight_y;
                }
            }
            output[row * 8 + column] = value;
        }
    }
}

static void thicken_pixels(const double *input, double *output) {
    for (int row = 0; row < 8; row++) {
        for (int column = 0; column < 8; column++) {
            int index = row * 8 + column;
            double neighbor = 0.0;
            if (column > 0 && input[index - 1] > neighbor) {
                neighbor = input[index - 1];
            }
            if (column < 7 && input[index + 1] > neighbor) {
                neighbor = input[index + 1];
            }
            if (row > 0 && input[index - 8] > neighbor) {
                neighbor = input[index - 8];
            }
            if (row < 7 && input[index + 8] > neighbor) {
                neighbor = input[index + 8];
            }
            double blended = input[index] * 0.88 + neighbor * 0.12;
            output[index] = input[index] > blended ? input[index] : blended;
            if (output[index] > 1.0) {
                output[index] = 1.0;
            }
        }
    }
}

WEB_EXPORT void web_cleanup(void) {
    free_net(model);
    model = NULL;
    clear_training_progress();
    free_dataset(&training_data);
    free_dataset(&test_data);
    free_dataset(&synthetic_data);
    memset(input_buffer, 0, sizeof(input_buffer));
    memset(prediction_probabilities, 0, sizeof(prediction_probabilities));
    memset(&latest_metrics, 0, sizeof(latest_metrics));
}

WEB_EXPORT int web_reset_model(unsigned int seed) {
    return replace_model(configured_hidden_sizes[0], configured_hidden_sizes[1],
                         seed);
}

WEB_EXPORT int web_configure_model(int first_hidden_size,
                                   int second_hidden_size,
                                   unsigned int seed) {
    if (first_hidden_size < MIN_HIDDEN_SIZE ||
        first_hidden_size > MAX_HIDDEN_SIZE ||
        second_hidden_size < MIN_HIDDEN_SIZE ||
        second_hidden_size > MAX_HIDDEN_SIZE) {
        return -1;
    }
    return replace_model(first_hidden_size, second_hidden_size, seed);
}

WEB_EXPORT int web_save_model(const char *path) {
    if (!path || !valid_web_model_architecture(model) ||
        training_order_ready || training_order_position != 0) {
        return -1;
    }
    return save_net(model, path);
}

WEB_EXPORT int web_load_model(const char *path) {
    if (!path || preflight_web_model_file(path) != 0) {
        return -1;
    }

    Net *candidate = load_net(path);
    if (!valid_web_model_architecture(candidate)) {
        free_net(candidate);
        return -1;
    }
    int snapshot_count = model_activation_count(candidate);
    double *candidate_snapshot = calloc((size_t)snapshot_count,
                                        sizeof(*candidate_snapshot));
    if (!candidate_snapshot) {
        free_net(candidate);
        return -1;
    }

    Net *previous_model = model;
    model = candidate;
    clear_training_progress();
    activation_snapshot = candidate_snapshot;
    activation_snapshot_count = snapshot_count;
    configured_hidden_sizes[0] = candidate->layer_sizes[1];
    configured_hidden_sizes[1] = candidate->layer_sizes[2];
    memset(input_buffer, 0, sizeof(input_buffer));
    memset(prediction_probabilities, 0, sizeof(prediction_probabilities));
    memset(&latest_metrics, 0, sizeof(latest_metrics));
    free_net(previous_model);
    return 0;
}

WEB_EXPORT int web_initialize(const char *training_path,
                              const char *test_path) {
    if (!training_path || !test_path) {
        return -1;
    }
    web_cleanup();
    if (load_optdigits_dataset(training_path, &training_data) != 0) {
        web_cleanup();
        return -1;
    }
    if (load_optdigits_dataset(test_path, &test_data) != 0) {
        web_cleanup();
        return -1;
    }
    if (web_reset_model(0x454e454eU) != 0) {
        web_cleanup();
        return -1;
    }
    return 0;
}

WEB_EXPORT int web_train_batch(int max_samples, double learning_rate) {
    if (!model || !training_data.inputs || !isfinite(learning_rate) ||
        learning_rate <= 0.0 || max_samples <= 0 ||
        prepare_training_order() != 0) {
        return -1;
    }

    int remaining = training_order_count - training_order_position;
    int processed = max_samples < remaining ? max_samples : remaining;
    double *last_input = NULL;
    double *last_target = NULL;
    for (int count = 0; count < processed; count++) {
        int sample = training_order[training_order_position++];
        last_input = combined_training_input(sample);
        last_target = combined_training_target(sample);
        forward_pass(last_input, model);
        backward_pass(last_target, model, learning_rate);
    }

    /* Show the updated model responding to the actual last trained sample. */
    forward_pass(last_input, model);
    memcpy(input_buffer, last_input, sizeof(input_buffer));
    last_training_label = target_label(last_target);
    capture_activations();
    memset(prediction_probabilities, 0, sizeof(prediction_probabilities));
    memset(&latest_metrics, 0, sizeof(latest_metrics));

    if (training_order_position == training_order_count) {
        completed_epochs++;
        training_order_position = 0;
        training_order_ready = 0;
    }
    return processed;
}

WEB_EXPORT int web_train_epoch(double learning_rate) {
    int starting_epoch = completed_epochs;
    int total_samples = training_data.num_samples + synthetic_data.num_samples;
    do {
        if (web_train_batch(total_samples, learning_rate) <= 0) {
            return -1;
        }
    } while (completed_epochs == starting_epoch);
    return 0;
}

WEB_EXPORT int web_evaluate(void) {
    if (!model || !test_data.inputs) {
        return -1;
    }
    latest_metrics = evaluate_classifier(test_data.inputs, test_data.targets,
                                         test_data.num_samples, model, NULL);
    return 0;
}

WEB_EXPORT double web_accuracy(void) {
    if (latest_metrics.total <= 0) {
        return 0.0;
    }
    return (double)latest_metrics.correct / latest_metrics.total;
}

WEB_EXPORT double web_loss(void) {
    return latest_metrics.cross_entropy;
}

WEB_EXPORT int web_training_samples(void) {
    return training_data.num_samples;
}

WEB_EXPORT int web_test_samples(void) {
    return test_data.num_samples;
}

WEB_EXPORT int web_epochs_trained(void) {
    return completed_epochs;
}

WEB_EXPORT int web_epoch_position(void) {
    return training_order_ready ? training_order_position : 0;
}

WEB_EXPORT int web_num_layers(void) {
    return model ? model->num_layers : 0;
}

WEB_EXPORT int web_layer_size(int layer) {
    if (!model || layer < 0 || layer >= model->num_layers) {
        return 0;
    }
    return model->layer_sizes[layer];
}

WEB_EXPORT double web_activation(int layer, int node) {
    if (!model || layer < 0 || layer >= model->num_layers || node < 0 ||
        node >= model->layer_sizes[layer]) {
        return 0.0;
    }
    return model->activations[layer][node];
}

WEB_EXPORT double *web_layer_weights(int layer) {
    if (!model || layer < 0 || layer >= model->num_layers - 1) {
        return NULL;
    }
    return model->weights[layer];
}

WEB_EXPORT double *web_activation_snapshot(void) {
    return activation_snapshot;
}

WEB_EXPORT int web_activation_count(void) {
    return activation_snapshot_count;
}

WEB_EXPORT int web_last_training_label(void) {
    return last_training_label;
}

WEB_EXPORT unsigned int web_activation_version(void) {
    return activation_version;
}

WEB_EXPORT int web_synthetic_samples(void) {
    return synthetic_data.num_samples;
}

WEB_EXPORT double *web_input_buffer(void) {
    return input_buffer;
}

WEB_EXPORT int web_inspect_input(void) {
    if (!model) {
        return -1;
    }
    for (int pixel = 0; pixel < INPUT_SIZE; pixel++) {
        if (!isfinite(input_buffer[pixel]) || input_buffer[pixel] < 0.0 ||
            input_buffer[pixel] > 1.0) {
            return -1;
        }
    }
    forward_pass(input_buffer, model);
    int output_layer = model->num_layers - 1;
    int prediction = 0;
    for (int digit = 1; digit < model->layer_sizes[output_layer]; digit++) {
        if (model->activations[output_layer][digit] >
            model->activations[output_layer][prediction]) {
            prediction = digit;
        }
    }
    return prediction;
}

WEB_EXPORT int web_predict(void) {
    if (!model) {
        return -1;
    }
    for (int pixel = 0; pixel < INPUT_SIZE; pixel++) {
        if (!isfinite(input_buffer[pixel]) || input_buffer[pixel] < 0.0 ||
            input_buffer[pixel] > 1.0) {
            return -1;
        }
    }

    const double offsets[PREDICTION_VARIANTS - 2][2] = {
        {-0.38, 0.0}, {0.38, 0.0}, {0.0, -0.34}, {0.0, 0.34},
    };
    double variant[INPUT_SIZE];
    memset(prediction_probabilities, 0, sizeof(prediction_probabilities));

    forward_pass(input_buffer, model);
    for (int digit = 0; digit < OUTPUT_SIZE; digit++) {
        prediction_probabilities[digit] =
            model->activations[model->num_layers - 1][digit] /
            PREDICTION_VARIANTS;
    }

    for (int index = 0; index < PREDICTION_VARIANTS - 2; index++) {
        translate_pixels(input_buffer, variant, offsets[index][0],
                         offsets[index][1]);
        forward_pass(variant, model);
        for (int digit = 0; digit < OUTPUT_SIZE; digit++) {
            prediction_probabilities[digit] +=
                model->activations[model->num_layers - 1][digit] /
                PREDICTION_VARIANTS;
        }
    }

    thicken_pixels(input_buffer, variant);
    forward_pass(variant, model);
    for (int digit = 0; digit < OUTPUT_SIZE; digit++) {
        prediction_probabilities[digit] +=
            model->activations[model->num_layers - 1][digit] /
            PREDICTION_VARIANTS;
    }

    int prediction = 0;
    for (int digit = 1; digit < OUTPUT_SIZE; digit++) {
        if (prediction_probabilities[digit] >
            prediction_probabilities[prediction]) {
            prediction = digit;
        }
    }
    return prediction;
}

WEB_EXPORT double web_probability(int digit) {
    if (digit < 0 || digit >= OUTPUT_SIZE) {
        return 0.0;
    }
    return prediction_probabilities[digit];
}

WEB_EXPORT int web_clear_synthetic_samples(void) {
    if (training_order_ready) {
        return -1;
    }
    free_dataset(&synthetic_data);
    return 0;
}

WEB_EXPORT int web_add_synthetic_sample(int label) {
    if (training_order_ready || label < 0 || label >= OUTPUT_SIZE) {
        return -1;
    }
    for (int pixel = 0; pixel < INPUT_SIZE; pixel++) {
        if (!isfinite(input_buffer[pixel]) || input_buffer[pixel] < 0.0 ||
            input_buffer[pixel] > 1.0) {
            return -1;
        }
    }

    int old_count = synthetic_data.num_samples;
    double **new_inputs = realloc(
        synthetic_data.inputs,
        sizeof(*synthetic_data.inputs) * (size_t)(old_count + 1));
    if (!new_inputs) {
        return -1;
    }
    synthetic_data.inputs = new_inputs;
    double **new_targets = realloc(
        synthetic_data.targets,
        sizeof(*synthetic_data.targets) * (size_t)(old_count + 1));
    if (!new_targets) {
        return -1;
    }
    synthetic_data.targets = new_targets;

    double *input = malloc(sizeof(*input) * INPUT_SIZE);
    double *target = calloc(OUTPUT_SIZE, sizeof(*target));
    if (!input || !target) {
        free(input);
        free(target);
        return -1;
    }
    memcpy(input, input_buffer, sizeof(*input) * INPUT_SIZE);
    target[label] = 1.0;
    synthetic_data.inputs[old_count] = input;
    synthetic_data.targets[old_count] = target;
    synthetic_data.num_samples++;
    synthetic_data.input_size = INPUT_SIZE;
    synthetic_data.num_classes = OUTPUT_SIZE;
    return 0;
}

WEB_EXPORT int web_copy_test_sample(int sample_index) {
    if (sample_index < 0 || sample_index >= test_data.num_samples) {
        return -1;
    }
    memcpy(input_buffer, test_data.inputs[sample_index], sizeof(input_buffer));
    return target_label(test_data.targets[sample_index]);
}

WEB_EXPORT int web_copy_training_sample(int sample_index) {
    int total_samples = training_data.num_samples + synthetic_data.num_samples;
    if (sample_index < 0 || sample_index >= total_samples) {
        return -1;
    }
    if (sample_index < training_data.num_samples) {
        memcpy(input_buffer, training_data.inputs[sample_index],
               sizeof(input_buffer));
        return target_label(training_data.targets[sample_index]);
    }
    int synthetic_index = sample_index - training_data.num_samples;
    memcpy(input_buffer, synthetic_data.inputs[synthetic_index],
           sizeof(input_buffer));
    return target_label(synthetic_data.targets[synthetic_index]);
}

WEB_EXPORT int web_test_label(int sample_index) {
    if (sample_index < 0 || sample_index >= test_data.num_samples) {
        return -1;
    }
    return target_label(test_data.targets[sample_index]);
}

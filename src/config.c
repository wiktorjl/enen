#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "config.h"

#define CONFIG_LINE_LENGTH 1024

enum ConfigField {
    FIELD_INPUT_SIZE = 1 << 0,
    FIELD_HIDDEN_LAYERS = 1 << 1,
    FIELD_OUTPUT_SIZE = 1 << 2,
    FIELD_LEARNING_RATE = 1 << 3,
    FIELD_EPOCHS = 1 << 4,
    FIELD_TRAIN_DATASET = 1 << 5,
    FIELD_TEST_DATASET = 1 << 6
};

#define REQUIRED_FIELDS ((1 << 7) - 1)

static char *trim(char *text) {
    while (isspace((unsigned char)*text)) {
        text++;
    }
    if (*text == '\0') {
        return text;
    }

    char *end = text + strlen(text) - 1;
    while (end > text && isspace((unsigned char)*end)) {
        end--;
    }
    end[1] = '\0';
    return text;
}

static int parse_positive_int(const char *text, int *value) {
    char *end = NULL;
    errno = 0;
    long parsed = strtol(text, &end, 10);
    while (end && isspace((unsigned char)*end)) {
        end++;
    }
    if (end == text || !end || *end != '\0' || errno == ERANGE ||
        parsed <= 0 || parsed > 1000000) {
        return -1;
    }
    *value = (int)parsed;
    return 0;
}

static int parse_learning_rate(const char *text, double *value) {
    char *end = NULL;
    errno = 0;
    double parsed = strtod(text, &end);
    while (end && isspace((unsigned char)*end)) {
        end++;
    }
    if (end == text || !end || *end != '\0' || errno == ERANGE ||
        !isfinite(parsed) || parsed <= 0.0 || parsed > 10.0) {
        return -1;
    }
    *value = parsed;
    return 0;
}

static int parse_hidden_layers(char *text, Config *config) {
    int count = 1;
    for (const char *cursor = text; *cursor; cursor++) {
        if (*cursor == ',') {
            count++;
        }
    }

    int *sizes = malloc(sizeof(*sizes) * (size_t)count);
    if (!sizes) {
        perror("Failed to allocate hidden-layer sizes");
        return -1;
    }

    int index = 0;
    char *cursor = text;
    while (cursor) {
        char *comma = strchr(cursor, ',');
        if (comma) {
            *comma = '\0';
        }
        char *value = trim(cursor);
        if (parse_positive_int(value, &sizes[index]) != 0) {
            free(sizes);
            return -1;
        }
        index++;
        cursor = comma ? comma + 1 : NULL;
    }

    config->hidden_layer_sizes = sizes;
    config->num_hidden_layers = count;
    return 0;
}

static int copy_path(char *destination, size_t destination_size,
                     const char *value) {
    size_t length = strlen(value);
    if (length == 0 || length >= destination_size) {
        return -1;
    }
    memcpy(destination, value, length + 1);
    return 0;
}

static int field_mask(const char *key) {
    if (strcmp(key, "input_size") == 0) return FIELD_INPUT_SIZE;
    if (strcmp(key, "hidden_layers") == 0) return FIELD_HIDDEN_LAYERS;
    if (strcmp(key, "output_size") == 0) return FIELD_OUTPUT_SIZE;
    if (strcmp(key, "learning_rate") == 0) return FIELD_LEARNING_RATE;
    if (strcmp(key, "epochs") == 0) return FIELD_EPOCHS;
    if (strcmp(key, "train_dataset") == 0) return FIELD_TRAIN_DATASET;
    if (strcmp(key, "test_dataset") == 0) return FIELD_TEST_DATASET;
    return 0;
}

static int store_field(Config *config, int field, char *value) {
    switch (field) {
        case FIELD_INPUT_SIZE:
            return parse_positive_int(value, &config->input_size);
        case FIELD_HIDDEN_LAYERS:
            return parse_hidden_layers(value, config);
        case FIELD_OUTPUT_SIZE:
            if (parse_positive_int(value, &config->output_size) != 0) return -1;
            return config->output_size >= 2 ? 0 : -1;
        case FIELD_LEARNING_RATE:
            return parse_learning_rate(value, &config->learning_rate);
        case FIELD_EPOCHS:
            return parse_positive_int(value, &config->epochs);
        case FIELD_TRAIN_DATASET:
            return copy_path(config->train_dataset_path,
                             sizeof(config->train_dataset_path), value);
        case FIELD_TEST_DATASET:
            return copy_path(config->test_dataset_path,
                             sizeof(config->test_dataset_path), value);
        default:
            return -1;
    }
}

Config *load_config(const char *filename) {
    if (!filename || filename[0] == '\0') {
        fprintf(stderr, "Configuration filename is empty\n");
        return NULL;
    }

    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Failed to open configuration '%s': ", filename);
        perror(NULL);
        return NULL;
    }

    Config *config = calloc(1, sizeof(*config));
    if (!config) {
        perror("Failed to allocate configuration");
        fclose(file);
        return NULL;
    }

    char line[CONFIG_LINE_LENGTH];
    int seen_fields = 0;
    int line_number = 0;
    int status = 0;
    while (fgets(line, sizeof(line), file)) {
        line_number++;
        if (!strchr(line, '\n') && !feof(file)) {
            fprintf(stderr, "%s:%d: configuration line is too long\n",
                    filename, line_number);
            status = -1;
            break;
        }

        char *content = trim(line);
        if (*content == '\0' || *content == '#') {
            continue;
        }
        char *equals = strchr(content, '=');
        if (!equals) {
            fprintf(stderr, "%s:%d: expected key=value\n", filename,
                    line_number);
            status = -1;
            break;
        }
        *equals = '\0';
        char *key = trim(content);
        char *value = trim(equals + 1);
        int field = field_mask(key);
        if (field == 0) {
            fprintf(stderr, "%s:%d: unknown configuration key '%s'\n",
                    filename, line_number, key);
            status = -1;
            break;
        }
        if (seen_fields & field) {
            fprintf(stderr, "%s:%d: duplicate configuration key '%s'\n",
                    filename, line_number, key);
            status = -1;
            break;
        }
        if (*value == '\0' || store_field(config, field, value) != 0) {
            fprintf(stderr, "%s:%d: invalid value for '%s'\n", filename,
                    line_number, key);
            status = -1;
            break;
        }
        seen_fields |= field;
    }

    if (ferror(file)) {
        fprintf(stderr, "Failed while reading configuration '%s'\n", filename);
        status = -1;
    }
    if (fclose(file) != 0) {
        fprintf(stderr, "Failed to close configuration '%s'\n", filename);
        status = -1;
    }
    if (status == 0 && seen_fields != REQUIRED_FIELDS) {
        fprintf(stderr, "Configuration '%s' is missing required fields\n",
                filename);
        status = -1;
    }
    if (status != 0) {
        free_config(config);
        return NULL;
    }
    return config;
}

void free_config(Config *config) {
    if (!config) {
        return;
    }
    free(config->hidden_layer_sizes);
    free(config);
}

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dataset.h"

#define INITIAL_CAPACITY 128
#define MAX_LINE_LENGTH 8192

static void clear_dataset(Dataset *dataset) {
    dataset->inputs = NULL;
    dataset->targets = NULL;
    dataset->num_samples = 0;
    dataset->input_size = 0;
    dataset->num_classes = 0;
}

void free_dataset(Dataset *dataset) {
    if (!dataset) {
        return;
    }

    for (int i = 0; i < dataset->num_samples; i++) {
        free(dataset->inputs[i]);
        free(dataset->targets[i]);
    }
    free(dataset->inputs);
    free(dataset->targets);
    clear_dataset(dataset);
}

static int grow_dataset(Dataset *dataset, int *capacity) {
    int new_capacity = *capacity * 2;
    double **new_inputs = realloc(
        dataset->inputs, sizeof(*dataset->inputs) * (size_t)new_capacity);
    if (!new_inputs) {
        return -1;
    }
    dataset->inputs = new_inputs;

    double **new_targets = realloc(
        dataset->targets, sizeof(*dataset->targets) * (size_t)new_capacity);
    if (!new_targets) {
        return -1;
    }
    dataset->targets = new_targets;
    *capacity = new_capacity;
    return 0;
}

static int parse_row(char *line, double *inputs, double *target,
                     int input_size, int num_classes, const char *filename,
                     int line_number) {
    char *cursor = line;

    for (int feature = 0; feature < input_size; feature++) {
        char *end = NULL;
        errno = 0;
        double value = strtod(cursor, &end);
        if (end == cursor || errno == ERANGE || !isfinite(value) ||
            value < 0.0 || value > 1.0) {
            fprintf(stderr,
                    "%s:%d: feature %d must be a number between 0 and 1\n",
                    filename, line_number, feature + 1);
            return -1;
        }
        while (isspace((unsigned char)*end)) {
            end++;
        }
        if (*end != ',') {
            fprintf(stderr, "%s:%d: expected a comma after feature %d\n",
                    filename, line_number, feature + 1);
            return -1;
        }
        inputs[feature] = value;
        cursor = end + 1;
    }

    char *end = NULL;
    errno = 0;
    long label = strtol(cursor, &end, 10);
    if (end == cursor || errno == ERANGE || label < 0 || label >= num_classes) {
        fprintf(stderr, "%s:%d: label must be an integer from 0 to %d\n",
                filename, line_number, num_classes - 1);
        return -1;
    }
    while (isspace((unsigned char)*end)) {
        end++;
    }
    if (*end != '\0') {
        fprintf(stderr, "%s:%d: unexpected data after the label\n",
                filename, line_number);
        return -1;
    }

    target[label] = 1.0;
    return 0;
}

int load_dataset(const char *filename, int input_size, int num_classes,
                 Dataset *dataset) {
    if (!filename || !dataset || input_size <= 0 || num_classes < 2) {
        fprintf(stderr, "Invalid dataset arguments\n");
        return -1;
    }

    clear_dataset(dataset);
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Failed to open dataset '%s': ", filename);
        perror(NULL);
        return -1;
    }

    int capacity = INITIAL_CAPACITY;
    dataset->inputs = calloc((size_t)capacity, sizeof(*dataset->inputs));
    dataset->targets = calloc((size_t)capacity, sizeof(*dataset->targets));
    if (!dataset->inputs || !dataset->targets) {
        perror("Failed to allocate dataset");
        fclose(file);
        free_dataset(dataset);
        return -1;
    }
    dataset->input_size = input_size;
    dataset->num_classes = num_classes;

    char line[MAX_LINE_LENGTH];
    int line_number = 0;
    int status = 0;
    while (fgets(line, sizeof(line), file)) {
        line_number++;
        if (!strchr(line, '\n') && !feof(file)) {
            fprintf(stderr, "%s:%d: line exceeds %d bytes\n", filename,
                    line_number, MAX_LINE_LENGTH - 1);
            status = -1;
            break;
        }

        char *cursor = line;
        while (isspace((unsigned char)*cursor)) {
            cursor++;
        }
        if (*cursor == '\0') {
            continue;
        }

        if (dataset->num_samples == capacity &&
            grow_dataset(dataset, &capacity) != 0) {
            perror("Failed to grow dataset");
            status = -1;
            break;
        }

        double *inputs = malloc(sizeof(*inputs) * (size_t)input_size);
        double *target = calloc((size_t)num_classes, sizeof(*target));
        if (!inputs || !target) {
            perror("Failed to allocate dataset row");
            free(inputs);
            free(target);
            status = -1;
            break;
        }
        if (parse_row(cursor, inputs, target, input_size, num_classes,
                      filename, line_number) != 0) {
            free(inputs);
            free(target);
            status = -1;
            break;
        }

        dataset->inputs[dataset->num_samples] = inputs;
        dataset->targets[dataset->num_samples] = target;
        dataset->num_samples++;
    }

    if (ferror(file)) {
        fprintf(stderr, "Failed while reading dataset '%s'\n", filename);
        status = -1;
    }
    if (fclose(file) != 0) {
        fprintf(stderr, "Failed to close dataset '%s'\n", filename);
        status = -1;
    }
    if (status == 0 && dataset->num_samples == 0) {
        fprintf(stderr, "Dataset '%s' contains no samples\n", filename);
        status = -1;
    }
    if (status != 0) {
        free_dataset(dataset);
    }
    return status;
}

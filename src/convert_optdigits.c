#include <ctype.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FEATURE_COUNT 64
#define MAX_LINE_LENGTH 4096

static int parse_row(char *line, int features[FEATURE_COUNT], int *label,
                     const char *input_path, size_t line_number) {
    char *cursor = line;

    for (int i = 0; i < FEATURE_COUNT; i++) {
        char *end = NULL;
        errno = 0;
        long value = strtol(cursor, &end, 10);

        if (end == cursor || errno == ERANGE || value < 0 || value > 16) {
            fprintf(stderr,
                    "%s:%zu: feature %d must be an integer from 0 to 16\n",
                    input_path, line_number, i + 1);
            return -1;
        }
        if (*end != ',') {
            fprintf(stderr, "%s:%zu: expected a comma after feature %d\n",
                    input_path, line_number, i + 1);
            return -1;
        }

        features[i] = (int)value;
        cursor = end + 1;
    }

    char *end = NULL;
    errno = 0;
    long parsed_label = strtol(cursor, &end, 10);
    if (end == cursor || errno == ERANGE || parsed_label < 0 || parsed_label > 9) {
        fprintf(stderr, "%s:%zu: label must be an integer from 0 to 9\n",
                input_path, line_number);
        return -1;
    }

    while (isspace((unsigned char)*end)) {
        end++;
    }
    if (*end != '\0') {
        fprintf(stderr, "%s:%zu: unexpected data after the label\n",
                input_path, line_number);
        return -1;
    }

    *label = (int)parsed_label;
    return 0;
}

static int convert_file(const char *input_path, const char *output_path) {
    if (strcmp(input_path, output_path) == 0) {
        fprintf(stderr, "Input and output paths must be different: %s\n", input_path);
        return -1;
    }

    FILE *input = fopen(input_path, "r");
    if (!input) {
        fprintf(stderr, "Failed to open input file '%s': ", input_path);
        perror(NULL);
        return -1;
    }

    FILE *output = fopen(output_path, "w");
    if (!output) {
        fprintf(stderr, "Failed to open output file '%s': ", output_path);
        perror(NULL);
        fclose(input);
        return -1;
    }

    char line[MAX_LINE_LENGTH];
    size_t line_number = 0;
    size_t sample_count = 0;
    int status = 0;

    while (fgets(line, sizeof(line), input)) {
        line_number++;

        if (!strchr(line, '\n') && !feof(input)) {
            fprintf(stderr, "%s:%zu: line is too long\n", input_path,
                    line_number);
            status = -1;
            break;
        }

        int features[FEATURE_COUNT];
        int label = 0;
        if (parse_row(line, features, &label, input_path, line_number) != 0) {
            status = -1;
            break;
        }

        for (int i = 0; i < FEATURE_COUNT; i++) {
            if (fprintf(output, "%.4f,", features[i] / 16.0) < 0) {
                status = -1;
                break;
            }
        }
        if (status != 0 || fprintf(output, "%d\n", label) < 0) {
            fprintf(stderr, "Failed while writing output file '%s'\n", output_path);
            status = -1;
            break;
        }

        sample_count++;
    }

    if (ferror(input)) {
        fprintf(stderr, "Failed while reading input file '%s'\n", input_path);
        status = -1;
    }
    if (status == 0 && sample_count == 0) {
        fprintf(stderr, "Input file '%s' contains no samples\n", input_path);
        status = -1;
    }
    if (fclose(input) != 0) {
        fprintf(stderr, "Failed to close input file '%s'\n", input_path);
        status = -1;
    }
    if (fclose(output) != 0) {
        fprintf(stderr, "Failed to close output file '%s'\n", output_path);
        status = -1;
    }

    if (status == 0) {
        printf("Converted %zu samples: %s -> %s\n", sample_count,
               input_path, output_path);
    } else {
        /* Never leave a partial CSV that could later be mistaken for valid data. */
        remove(output_path);
    }
    return status;
}

static void print_usage(const char *program_name) {
    fprintf(stderr,
            "Usage: %s INPUT OUTPUT [INPUT OUTPUT ...]\n\n"
            "Convert UCI optdigits rows to normalized CSV rows containing\n"
            "64 floating-point features followed by the digit label.\n\n"
            "Example:\n"
            "  %s datasets/optdigits.tra datasets/UCI_digits_train.csv "
            "datasets/optdigits.tes datasets/UCI_digits_test.csv\n",
            program_name, program_name);
}

int main(int argc, char *argv[]) {
    if (argc < 3 || argc % 2 == 0) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    for (int i = 1; i < argc; i += 2) {
        if (convert_file(argv[i], argv[i + 1]) != 0) {
            return EXIT_FAILURE;
        }
    }

    return EXIT_SUCCESS;
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <limits.h>
#include "config.h"
#include "tools.h"


int validate_filename(const char * filename) {
    if(!filename) {
        return 0;
    } else if(filename[0] == '\0') {
        return 0;
    }

    return 1;
}

static int is_absolute_path(const char *path) {
    return path && path[0] == '/';
}

static int has_path_separator(const char *path) {
    return path && strchr(path, '/') != NULL;
}

static int get_executable_dir(char *out_dir, size_t out_dir_size) {
    if (!out_dir || out_dir_size == 0) {
        return 0;
    }

    ssize_t len = readlink("/proc/self/exe", out_dir, out_dir_size - 1);
    if (len <= 0 || (size_t)len >= out_dir_size) {
        return 0;
    }

    out_dir[len] = '\0';
    char *last_slash = strrchr(out_dir, '/');
    if (!last_slash) {
        return 0;
    }
    *last_slash = '\0';
    return 1;
}

static FILE *open_config_with_fallback(const char *filename, char *resolved_path, size_t resolved_path_size) {
    FILE *f = fopen(filename, "r");
    if (f) {
        strncpy(resolved_path, filename, resolved_path_size - 1);
        resolved_path[resolved_path_size - 1] = '\0';
        return f;
    }

    // If a path was provided, don't try to reinterpret it.
    if (has_path_separator(filename)) {
        return NULL;
    }

    char exe_dir[PATH_MAX];
    if (!get_executable_dir(exe_dir, sizeof(exe_dir))) {
        return NULL;
    }

    char candidate[PATH_MAX];
    int written = snprintf(candidate, sizeof(candidate), "%s/%s", exe_dir, filename);
    if (written <= 0 || written >= (int)sizeof(candidate)) {
        return NULL;
    }

    f = fopen(candidate, "r");
    if (!f) {
        return NULL;
    }

    strncpy(resolved_path, candidate, resolved_path_size - 1);
    resolved_path[resolved_path_size - 1] = '\0';
    return f;
}

Config* load_config(const char* filename) {

    Config* config = malloc(sizeof(Config));
    if(!config) {
        perror("Failed to allocate memory for config");
        return NULL;
    }
    memset(config, 0, sizeof(Config));


    // Check if filename is non empty
    if(!validate_filename(filename)) {
        perror("Filename invalid.");
        free(config);
        return NULL;
    }

    char config_path[PATH_MAX] = {0};
    FILE * f = open_config_with_fallback(filename, config_path, sizeof(config_path));

    if (!f) {
        perror("Failed to open config file");
        free(config);
        return NULL;
    } else {
        char line[1000];
        char config_dir[PATH_MAX] = ".";
        char *last_slash = strrchr(config_path, '/');
        if (last_slash) {
            *last_slash = '\0';
            strncpy(config_dir, config_path, sizeof(config_dir) - 1);
            config_dir[sizeof(config_dir) - 1] = '\0';
        }

        while (fgets(line, sizeof(line), f)) {
            char line_trimmed[1000];
            trim_copy(line, line_trimmed, sizeof(line_trimmed));

            if (line_trimmed[0] == '\0' || line_trimmed[0] == '#') {
                continue;
            }

            char *key_raw = strtok(line_trimmed, "=");
            char *val_raw = strtok(NULL, "=");
            if (!key_raw || !val_raw) {
                continue;
            }

            char key[1000];
            char val[1000];
            trim_copy(key_raw, key, sizeof(key));
            trim_copy(val_raw, val, sizeof(val));

            if(strcmp(key, "input_size") == 0) {
                config->input_size = atoi(val);
            } else if(strcmp(key, "hidden_layers") == 0) {
                // Parse comma-separated sizes
                char layers_buf[1000];
                strncpy(layers_buf, val, sizeof(layers_buf) - 1);
                layers_buf[sizeof(layers_buf) - 1] = '\0';

                char *layer_size_str = strtok(layers_buf, ",");
                int layer_count = 0;
                int *layer_sizes = NULL;

                while (layer_size_str) {
                    char layer_size_trimmed[1000];
                    trim_copy(layer_size_str, layer_size_trimmed, sizeof(layer_size_trimmed));
                    layer_sizes = realloc(layer_sizes, sizeof(int) * (layer_count + 1));
                    layer_sizes[layer_count] = atoi(layer_size_trimmed);
                    layer_count++;
                    layer_size_str = strtok(NULL, ",");
                }

                free(config->hidden_layer_sizes);
                config->num_hidden_layers = layer_count;
                config->hidden_layer_sizes = layer_sizes;
            } else if(strcmp(key, "output_size") == 0) {
                config->output_size = atoi(val);
            } else if(strcmp(key, "learning_rate") == 0) {
                config->learning_rate = atof(val);
            } else if(strcmp(key, "epochs") == 0) {
                config->epochs = atoi(val);
            } else if(strcmp(key, "dataset") == 0) {
                if (!is_absolute_path(val)) {
                    char dataset_resolved[PATH_MAX];
                    int written = snprintf(dataset_resolved, sizeof(dataset_resolved), "%s/%s", config_dir, val);
                    if (written > 0 && written < (int)sizeof(dataset_resolved)) {
                        strncpy(config->dataset_path, dataset_resolved, sizeof(config->dataset_path) - 1);
                        config->dataset_path[sizeof(config->dataset_path) - 1] = '\0';
                    } else {
                        config->dataset_path[0] = '\0';
                    }
                } else {
                    strncpy(config->dataset_path, val, sizeof(config->dataset_path) - 1);
                    config->dataset_path[sizeof(config->dataset_path) - 1] = '\0';
                }
            } else {
                printf("Unknown key: %s\n", key);
                printf("Storing config unknown_key=%s\n", val);
            }
        }
    }
    fclose(f);
    return config;
}

void free_config(Config* config) {
    // todo: drill into struct and free elements
    free(config->hidden_layer_sizes);
    // free(config->dataset_path);
    free(config);
}

void print_config(const Config* config) {
    if(!config) {
        printf("Config is NULL\n");
        return;
    }
    printf("Config:\n");
    printf("  Input Size: %d\n", config->input_size);
    printf("  Output Size: %d\n", config->output_size);
    printf("  Num Hidden Layers: %d\n", config->num_hidden_layers);
    for(int i = 0; i < config->num_hidden_layers; i++) {
        printf("    Hidden Layer %d Size: %d\n", i + 1, config->hidden_layer_sizes[i]);
    }
    printf("  Learning Rate: %f\n", config->learning_rate);
    printf("  Epochs: %d\n", config->epochs);
    printf("  Dataset Path: %s\n", config->dataset_path);
}

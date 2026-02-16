#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
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

/*
 * load_config() - Load network configuration from file
 *
 * Parses a simple key=value format configuration file.
 * Required fields: input_size, hidden_layers, output_size, learning_rate, epochs, dataset
 *
 * Returns:
 *   Pointer to Config structure, or NULL on error
 */
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
        return NULL;
    }
    // printf("Filename is valid.\n");
    
    // Check if file exists and is readable
    if(access(filename, R_OK) != 0) {
        perror("File is not accessible");
        return NULL;
    }

    // Read into memory
    FILE * f = fopen(filename, "r");

    if (!f) {
        perror("Failed to open config file");
        return NULL;
    } else {
        char line[1000];
        char * res = NULL;

        
        do {
            res = fgets(line, 1000, f);
            char line_trimmed[1000];
            trim_copy(line, line_trimmed, 1000);

            if(res) {
                // printf("Line: %s\n", line_trimmed);
                char * key = strtok(line_trimmed, "=");
                char * val = strtok(NULL, "=");

                // Skip lines without '=' or with empty key/value
                if (!key || !val || strlen(key) == 0 || strlen(val) == 0) {
                    continue;
                }

                if(strcmp(key, "input_size") == 0) {
                    // printf("Storing config input_size=%s\n", val);
                    config->input_size = atoi(val);
                } else if(strcmp(key, "hidden_layers") == 0) {
                    // printf("Storing config hidden_layers=%s\n", val);
                    // Parse comma-separated sizes
                    char * layer_size_str = strtok(val, ",");
                    int layer_count = 0;
                    int * layer_sizes = NULL;
                    while(layer_size_str) {
                        layer_sizes = realloc(layer_sizes, sizeof(int) * (layer_count + 1));
                        layer_sizes[layer_count] = atoi(layer_size_str);
                        layer_count++;
                        layer_size_str = strtok(NULL, ",");
                    }
                    config->num_hidden_layers = layer_count;
                    config->hidden_layer_sizes = layer_sizes;
                     

                } else if(strcmp(key, "output_size") == 0) {
                    // printf("Storing config output_size=%s\n", val);
                    config->output_size = atoi(val);
                } else if(strcmp(key, "learning_rate") == 0) {
                    // printf("Storing config learning_rate=%s\n", val);
                    config->learning_rate = atof(val);
                } else if(strcmp(key, "epochs") == 0) {
                    // printf("Storing config epochs=%s\n", val);
                    config->epochs = atoi(val);
                } else if(strcmp(key, "dataset") == 0) {
                    // printf("Storing config dataset=%s\n", val);
                    if(val) {
                        // Copy dataset path safely into fixed-size buffer
                        strncpy(config->dataset_path, val, sizeof(config->dataset_path) - 1);
                        config->dataset_path[sizeof(config->dataset_path) - 1] = '\0';
                    }
                } else {
                    fprintf(stderr, "Warning: Unknown configuration key '%s' (ignored)\n", key);
                }
            }
        } while(res);
    }
    fclose(f);

    // Validate required fields
    if (config->input_size <= 0) {
        fprintf(stderr, "Error: Invalid or missing 'input_size' in config\n");
        free_config(config);
        return NULL;
    }
    if (config->output_size <= 0) {
        fprintf(stderr, "Error: Invalid or missing 'output_size' in config\n");
        free_config(config);
        return NULL;
    }
    if (config->num_hidden_layers < 0) {
        fprintf(stderr, "Error: Invalid 'hidden_layers' in config\n");
        free_config(config);
        return NULL;
    }
    if (config->learning_rate <= 0.0 || config->learning_rate > 10.0) {
        fprintf(stderr, "Error: Invalid 'learning_rate' %.2f (expected 0.0 < rate <= 10.0)\n",
                config->learning_rate);
        free_config(config);
        return NULL;
    }
    if (config->epochs <= 0) {
        fprintf(stderr, "Error: Invalid or missing 'epochs' in config\n");
        free_config(config);
        return NULL;
    }
    if (strlen(config->dataset_path) == 0) {
        fprintf(stderr, "Error: Missing 'dataset' path in config\n");
        free_config(config);
        return NULL;
    }

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
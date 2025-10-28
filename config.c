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
                // printf("Key=%s, Val=%s\n", key, val);

                if(strcmp(key, "input_size") == 0) {
                    // printf("Storing config input_size=%s\n", val);
                    config->input_size = atoi(val);
                } else if(strcmp(key, "hidden_layers") == 0) {
                    // printf("Storing config hidden_layers=%s\n", val);
                    config->num_hidden_layers = atoi(val);
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
                    printf("Unknown key: %s\n", key);
                    printf("Storing cconfig unknown_key=%s\n", val);
                }
            }
        } while(res);
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
    printf("  Learning Rate: %f\n", config->learning_rate);
    printf("  Epochs: %d\n", config->epochs);
    printf("  Dataset Path: %s\n", config->dataset_path);
}
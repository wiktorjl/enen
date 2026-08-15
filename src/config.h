#ifndef CONFIG_H
#define CONFIG_H

typedef struct {
    int input_size;
    int output_size;
    int num_hidden_layers;
    int *hidden_layer_sizes;  // Array of sizes for each hidden layer
    double learning_rate;
    int epochs;
    char train_dataset_path[256];
    char test_dataset_path[256];
} Config;

Config* load_config(const char* filename);
void free_config(Config* config);

#endif // CONFIG_H

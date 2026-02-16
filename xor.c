#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "nn.h"
#include "config.h"
#include "tools.h"

int main() {
    Config* config = load_config("xornet.conf");
    if(!config) {
        fprintf(stderr, "Failed to load config file.\n");
        return 1;
    }

    print_config(config);

    srand(time(NULL));

    Net *net = create_net(config);
    if (!net) {
        fprintf(stderr, "Failed to create network.\n");
        free_config(config);
        return 1;
    }

    double **inputs = NULL;
    double *expected = NULL;
    int num_samples = 0;

    load_dataset(config->dataset_path, &inputs, &expected, &num_samples, config->input_size);

    printf("\nTraining network with %d samples...\n", num_samples);
    train_nn(inputs, expected, num_samples, net, config->epochs, config->learning_rate);

    test_nn(inputs, expected, num_samples, net);

    double mse = test_nn_and_get_mse(inputs, expected, num_samples, net);
    printf("\nFinal MSE: %.6f\n", mse);

    printf("\nSaving model to models/xor.model...\n");
    if (save_net(net, "models/xor.model") == 0) {
        printf("Model saved successfully\n");
    } else {
        fprintf(stderr, "Failed to save model\n");
    }

    free_dataset(inputs, expected, num_samples);
    free_net(net);
    free_config(config);
    return 0;
}

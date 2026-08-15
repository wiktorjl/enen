#ifndef DATASET_H
#define DATASET_H

typedef struct {
    double **inputs;
    double **targets;
    int num_samples;
    int input_size;
    int num_classes;
} Dataset;

/* Load normalized features and integer class labels from a CSV file. */
int load_dataset(const char *filename, int input_size, int num_classes,
                 Dataset *dataset);
void free_dataset(Dataset *dataset);

#endif

#ifndef DATASET_H
#define DATASET_H

typedef struct {
    double **inputs;
    double **targets;
    int num_samples;
    int input_size;
    int num_classes;
} Dataset;

/* Load normalized features and integer class labels from a CSV file.
 * `dataset` must be zero-initialized on first use. Loading again replaces and
 * frees its existing contents. */
int load_dataset(const char *filename, int input_size, int num_classes,
                 Dataset *dataset);

/* Load the original UCI optical-digits format (64 integer pixels in [0, 16]
 * followed by a label) and normalize each pixel to [0, 1]. The same
 * initialization and replacement rule as load_dataset() applies. */
int load_optdigits_dataset(const char *filename, Dataset *dataset);
void free_dataset(Dataset *dataset);

#endif

#ifndef NN_H
#define NN_H

#include "config.h"

typedef struct {
    int num_layers;
    int *layer_sizes;
    double **weights;      /* weights[layer][source * next_size + destination] */
    double **biases;       /* One vector per connection, for its destination. */
    double **activations;  /* One vector per layer; layer zero copies the input. */
    double **deltas;       /* Negative loss gradients wrt. preactivations. */
} Net;

typedef struct {
    int correct;
    int total;
    double cross_entropy;
} ClassificationMetrics;

Net *create_net(const Config *config);
Net *clone_net(const Net *source);
void free_net(Net *net);
void print_net(const Net *net);
int net_matches_dimensions(const Net *net, int input_size, int num_classes);

/* All buffers must match the network's dimensions. Forward propagation copies
 * inputs and computes sigmoid hidden activations and softmax probabilities. */
void forward_pass(const double *inputs, Net *net);
/* Learn from the immediately preceding forward pass. target is one-hot;
 * learning_rate is finite and positive. Activations describe that example. */
void backward_pass(const double *target, Net *net, double learning_rate);
int predict_class(const double *inputs, Net *net);
int train_classifier(double **inputs, double **targets, int num_samples,
                     Net *net, int epochs, double learning_rate);
ClassificationMetrics evaluate_classifier(double **inputs, double **targets,
                                            int num_samples, Net *net,
                                            int *confusion_matrix);
void print_confusion_matrix(const int *confusion_matrix, int num_classes);

/* Saving replaces filename only after the complete new model is written. */
int save_net(const Net *net, const char *filename);
Net *load_net(const char *filename);

#endif

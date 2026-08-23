#ifndef WEB_API_H
#define WEB_API_H

/* Small C boundary used by the Emscripten browser build. The browser writes
 * 64 normalized pixels into web_input_buffer(), and every model operation
 * below runs through the same nn.c implementation as the native programs. */
int web_initialize(const char *training_path, const char *test_path);
int web_reset_model(unsigned int seed);
int web_configure_model(int first_hidden_size, int second_hidden_size,
                        unsigned int seed);
int web_save_model(const char *path);
int web_load_model(const char *path);
int web_train_batch(int max_samples, double learning_rate);
int web_train_epoch(double learning_rate);
int web_evaluate(void);

double web_accuracy(void);
double web_loss(void);
int web_training_samples(void);
int web_test_samples(void);
int web_epochs_trained(void);
int web_epoch_position(void);
int web_synthetic_samples(void);
int web_num_layers(void);
int web_layer_size(int layer);
double web_activation(int layer, int node);
double *web_layer_weights(int layer);
double *web_activation_snapshot(void);
int web_activation_count(void);
int web_last_training_label(void);
unsigned int web_activation_version(void);

double *web_input_buffer(void);
int web_inspect_input(void);
int web_predict(void);
double web_probability(int digit);

int web_clear_synthetic_samples(void);
int web_add_synthetic_sample(int label);

int web_copy_test_sample(int sample_index);
int web_copy_training_sample(int sample_index);
int web_test_label(int sample_index);
void web_cleanup(void);

#endif

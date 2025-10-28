#ifndef TOOLS_H
#define TOOLS_H



typedef struct NetStruct {
    double weights_ih[2][2];
    double weights_ho[2][1];

    double bias_hidden[2];
    double bias_output[1];

    double output_hidden[2];
    double output_final[1];
} Net;



void print_net(const Net *net, int verbose);
double randinit();
void init_net(Net *net);
double sigmoid(double input);
double sigmoid_derivative(double sigmoid_output);

void shuffle_array(int n, double *arr);
void init_xor_data(const char *filename, double inputs[4][2], double expected[4]);
int * init_order_array(int n);

char *trim_copy(char *src, char *dest, int destsize);
#endif // TOOLS_H
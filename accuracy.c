#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "nn.h"

// --- ASCII VISUALIZATION HELPERS ---

// Prints a single ASCII bar for a plot or histogram
void print_bar(double value, double max_value, int width) {
    if (value < 0) value = 0;
    int bar_length = (int)((value / max_value) * width);
    if (bar_length < 0) bar_length = 0;
    if (bar_length > width) bar_length = width;
    
    printf("[");
    for (int i = 0; i < width; ++i) {
        if (i < bar_length) {
            printf("#");
        } else {
            printf(" ");
        }
    }
    printf("]");
}

// Prints a histogram for the discrete accuracy values (0, 25, 50, 75, 100)
void print_histogram(double* accuracies, int num_runs) {
    int counts[5] = {0}; // Bins for 0, 25, 50, 75, 100%

    for (int i = 0; i < num_runs; i++) {
        int acc_val = (int)round(accuracies[i]);
        if (acc_val == 0) counts[0]++;
        else if (acc_val == 25) counts[1]++;
        else if (acc_val == 50) counts[2]++;
        else if (acc_val == 75) counts[3]++;
        else if (acc_val == 100) counts[4]++;
    }

    int max_count = 0;
    for (int i = 0; i < 5; i++) {
        if (counts[i] > max_count) {
            max_count = counts[i];
        }
    }

    printf("\n--- Accuracy Distribution Histogram ---\n");
    if (max_count == 0) {
        printf("No data to display.\n");
        return;
    }

    double levels[] = {0.0, 25.0, 50.0, 75.0, 100.0};
    for (int i = 0; i < 5; i++) {
        printf("%5.1f%% | ", levels[i]);
        print_bar(counts[i], max_count, 50);
        printf(" (%d runs)\n", counts[i]);
    }
}


// --- MAIN APPLICATION ---

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <number_of_runs>\n", argv[0]);
        return 1;
    }

    int num_runs = atoi(argv[1]);
    if (num_runs <= 0) {
        fprintf(stderr, "Number of runs must be a positive integer.\n");
        return 1;
    }

    double inputs[4][2];
    double expected[4];
    init_xor_data("xor_dataset.csv", inputs, expected);

    double* accuracies = malloc(num_runs * sizeof(double));
    if (!accuracies) {
        perror("Failed to allocate memory for accuracies");
        return 1;
    }

    srand(time(NULL));
    
    printf("--- Running %d training sessions... ---\n\n", num_runs);
    printf("--- Live Average Accuracy Plateau ---\n");

    double running_total_accuracy = 0.0;

    for (int run = 0; run < num_runs; run++) {
        Net *net = malloc(sizeof(Net));
        init_net(net);

        train_nn(inputs, expected, net, 10000, 0.5);

        int correct_predictions = 0;
        for (int i = 0; i < 4; i++) {
            forward_pass(inputs[i][0], inputs[i][1], net);
            int prediction = (net->output_final[0] > 0.5) ? 1 : 0;
            if (prediction == (int)expected[i]) {
                correct_predictions++;
            }
        }
        accuracies[run] = (double)correct_predictions / 4.0 * 100.0;
        running_total_accuracy += accuracies[run];
        double running_avg = running_total_accuracy / (run + 1);

        free(net);
        
        // Live Plateau Plot
        printf("Run %4d/%-4d | Acc: %5.1f%% | Avg: %5.1f%% ", run + 1, num_runs, accuracies[run], running_avg);
        print_bar(running_avg, 100, 40);
        printf("\n");
    }

    // --- Final Summary Calculation ---
    double min_accuracy = 100.0;
    double max_accuracy = 0.0;
    
    for (int i = 0; i < num_runs; i++) {
        if (accuracies[i] < min_accuracy) min_accuracy = accuracies[i];
        if (accuracies[i] > max_accuracy) max_accuracy = accuracies[i];
    }
    double final_avg_accuracy = running_total_accuracy / num_runs;

    // Standard Deviation
    double sum_sq_diff = 0.0;
    for (int i = 0; i < num_runs; i++) {
        sum_sq_diff += pow(accuracies[i] - final_avg_accuracy, 2);
    }
    double std_dev = (num_runs > 1) ? sqrt(sum_sq_diff / (num_runs - 1)) : 0.0;

    // Standard Error of the Mean
    double sem = (num_runs > 0) ? std_dev / sqrt(num_runs) : 0.0;

    printf("\n\n--- Statistical Summary over %d runs ---\
", num_runs);
    printf("Average Accuracy: %6.2f%%\n", final_avg_accuracy);
    printf("Standard Deviation: %6.2f%%\n", std_dev);
    printf("Standard Error:   %6.2f%%\n", sem);
    printf("Minimum Accuracy: %6.2f%%\n", min_accuracy);
    printf("Maximum Accuracy: %6.2f%%\n", max_accuracy);
    
    // ASCII Histogram
    print_histogram(accuracies, num_runs);

    free(accuracies);

    return 0;
}

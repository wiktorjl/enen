# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a simple neural network implementation in C that learns the XOR function. The network uses backpropagation with a 2-2-1 architecture (2 input neurons, 2 hidden neurons, 1 output neuron).

## Architecture

The codebase is organized into three main files:

- **xor.c**: Contains the main program, training loop, and neural network forward/backward pass logic
- **tools.h**: Defines the `Net` struct and function declarations for network utilities
- **tools.c**: Implements utility functions for network initialization, sigmoid activation, data loading, and array shuffling
- **xor_dataset.csv**: Training data in format `input1,input2,expected_output`

### Key Components

**Net struct** (tools.h:6-15): The core neural network structure containing:
- `weights_ih[2][2]`: Input-to-hidden layer weights
- `weights_ho[2][1]`: Hidden-to-output layer weights
- `bias_hidden[2]`: Hidden layer biases
- `bias_output[1]`: Output layer bias
- `output_hidden[2]`: Hidden layer activations (cached during forward pass)
- `output_final[1]`: Final output value

**Training process** (xor.c:55-74):
- Runs for `ROUNDS` (10000) iterations
- Shuffles input order each round to prevent bias
- Performs forward pass followed by backward pass for gradient updates
- Uses learning rate of 0.5 (hardcoded in backward_pass)

## Building and Running

Compile the program:
```bash
gcc -o xor xor.c tools.c -lm
```

Run the trained network:
```bash
./xor
```

The program will train on XOR data and print the learned outputs for all 4 input combinations.

## Development Notes

- The network uses sigmoid activation: `sigmoid(x) = 1/(1 + e^(-x))`
- Weights are initialized randomly in range [-1, 1] via `randinit()`
- Training data is loaded from CSV file via `init_xor_data()`
- Memory leak: `order` array allocated in training loop (xor.c:62) is never freed

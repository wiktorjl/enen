# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a configurable neural network implementation in C that can learn various functions including XOR. The network uses backpropagation with a fully configurable multi-layer architecture.

## Architecture

The codebase is organized into several files:

- **xor.c**: Main program that trains and tests the network on XOR
- **gym.c**: Hyperparameter tuning tool that tests different learning rates and training rounds
- **accuracy.c**: Statistical analysis tool that runs multiple training sessions and reports accuracy metrics
- **nn.h/nn.c**: Core neural network implementation with forward/backward passes
- **config.h/config.c**: Configuration file parser for network architecture and training parameters
- **tools.h/tools.c**: Utility functions for dataset loading, activation functions, and network visualization
- **xornet.conf**: Configuration file specifying network architecture and training parameters
- **xor_dataset.csv**: Training data in CSV format

### Key Components

**Config struct** (config.h:4-12): Configuration parameters loaded from file:
- `input_size`: Number of input neurons
- `output_size`: Number of output neurons
- `num_hidden_layers`: Number of hidden layers
- `hidden_layer_sizes`: Array of hidden layer sizes
- `learning_rate`: Training learning rate
- `epochs`: Number of training iterations
- `dataset_path`: Path to CSV dataset

**Net struct** (nn.h:6-12): The dynamic neural network structure:
- `num_layers`: Total number of layers (input + hidden + output)
- `layer_sizes`: Array of neuron counts for each layer
- `weights`: Dynamically allocated weight matrices between layers
- `biases`: Bias values for each layer
- `activations`: Cached activation values during forward pass

**Training process** (nn.c:135-148):
- Configurable number of epochs from config file
- Shuffles input order each round to prevent bias
- Performs forward pass through all layers
- Backpropagates error from output to input
- Configurable learning rate from config file

## Building and Running

Build all programs:
```bash
make
```

Train and test on XOR:
```bash
./xor
```

Run hyperparameter tuning:
```bash
./gym
```

Run statistical accuracy analysis:
```bash
./accuracy 100  # Run 100 training sessions
```

## Configuration File Format

The `xornet.conf` file uses a simple key=value format:
```
input_size=2
hidden_layers=3,4,3
output_size=1
learning_rate=0.5
epochs=10000
dataset=xor_dataset.csv
```

## Development Notes

- The network uses sigmoid activation: `sigmoid(x) = 1/(1 + e^(-x))`
- Weights are initialized randomly in range [-1, 1] via `randinit()`
- Network architecture is fully dynamic and configurable
- Memory is properly managed with `free_net()`, `free_config()`, and `free_dataset()`
- Dataset format is CSV: input values followed by expected output

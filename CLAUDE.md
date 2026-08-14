# CLAUDE.md

Guidance for working with this repository.

## Project overview

This is a C implementation of a configurable 8×8 handwritten-digit classifier.
It uses a fully connected network with sigmoid activations, one-hot targets,
backpropagation, stochastic gradient descent, and OpenMP.

## Main components

- `src/digits.c`: train, evaluate, save, and load the digit model
- `src/gym.c`: compare learning-rate and epoch combinations
- `src/accuracy.c`: repeat training and summarize test accuracy
- `src/nn.c` / `src/nn.h`: network lifecycle, propagation, training, and persistence
- `src/config.c` / `src/config.h`: configuration parsing
- `src/tools.c` / `src/tools.h`: multiclass CSV loading and numerical helpers
- `conf/digits.conf`: architecture, training parameters, and dataset paths

## Build and run

```bash
make all
./build/digits
./build/digits --load models/digits.model
./build/gym
./build/accuracy 10
```

## Data and evaluation

CSV rows contain 64 normalized pixels followed by an integer label from 0 to 9.
`load_dataset_multiclass()` converts labels to one-hot vectors. Training must use
`train_dataset_path`; metrics must use `test_dataset_path`. Predictions use the
argmax across all output neurons.

## Development notes

- Preserve the dynamic architecture represented by `Config` and `Net`.
- Keep model and dataset dimensions consistent with `conf/digits.conf`.
- Use `free_dataset_multiclass()`, `free_net()`, and `free_config()` for cleanup.
- Build with `make` so GCC warnings and OpenMP flags match the project.

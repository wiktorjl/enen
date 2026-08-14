# GEMINI.md

Guidance for working with this repository.

## Project overview

The project is an educational handwritten-digit classifier written in C. The
default network accepts 64 normalized pixels, uses two configurable hidden
layers, and produces ten class activations.

## Code organization

- `src/digits.c`: main training and evaluation workflow
- `src/gym.c`: hyperparameter experiments using train/test splits
- `src/accuracy.c`: repeated training runs and accuracy statistics
- `src/nn.c` / `src/nn.h`: neural-network implementation and model persistence
- `src/tools.c` / `src/tools.h`: one-hot dataset loading and helpers
- `src/config.c` / `src/config.h`: configuration parsing
- `conf/digits.conf`: active network and dataset configuration

## Build and run

```bash
make all
./build/digits
./build/digits --load models/digits.model
```

## Important conventions

- Dataset rows contain 64 pixel values followed by a digit label.
- Labels are represented as ten-element one-hot vectors.
- Training uses the configured training split; evaluation uses the test split.
- Predictions are the argmax across all ten output activations.
- Dynamically allocated networks, configurations, and datasets must be released
  with their corresponding cleanup functions.

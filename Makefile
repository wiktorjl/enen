# Compiler and flags
CC = gcc
CFLAGS = -g -Wall -fopenmp
LDFLAGS = -lm -fopenmp

# Directories
BUILD_DIR = build
MODELS_DIR = models

# All executables
all: directories xor gym accuracy digits generate_digits

# Create necessary directories
directories:
	@mkdir -p $(BUILD_DIR) $(MODELS_DIR)

# Executable rules
xor: $(BUILD_DIR)/xor.o $(BUILD_DIR)/nn.o $(BUILD_DIR)/tools.o $(BUILD_DIR)/config.o
	$(CC) $(CFLAGS) -o $(BUILD_DIR)/xor $(BUILD_DIR)/xor.o $(BUILD_DIR)/nn.o $(BUILD_DIR)/tools.o $(BUILD_DIR)/config.o $(LDFLAGS)

gym: $(BUILD_DIR)/gym.o $(BUILD_DIR)/nn.o $(BUILD_DIR)/tools.o $(BUILD_DIR)/config.o
	$(CC) $(CFLAGS) -o $(BUILD_DIR)/gym $(BUILD_DIR)/gym.o $(BUILD_DIR)/nn.o $(BUILD_DIR)/tools.o $(BUILD_DIR)/config.o $(LDFLAGS)

accuracy: $(BUILD_DIR)/accuracy.o $(BUILD_DIR)/nn.o $(BUILD_DIR)/tools.o $(BUILD_DIR)/config.o
	$(CC) $(CFLAGS) -o $(BUILD_DIR)/accuracy $(BUILD_DIR)/accuracy.o $(BUILD_DIR)/nn.o $(BUILD_DIR)/tools.o $(BUILD_DIR)/config.o $(LDFLAGS)

digits: $(BUILD_DIR)/digits.o $(BUILD_DIR)/nn.o $(BUILD_DIR)/tools.o $(BUILD_DIR)/config.o
	$(CC) $(CFLAGS) -o $(BUILD_DIR)/digits $(BUILD_DIR)/digits.o $(BUILD_DIR)/nn.o $(BUILD_DIR)/tools.o $(BUILD_DIR)/config.o $(LDFLAGS)

generate_digits: generate_digits.c
	$(CC) $(CFLAGS) -o $(BUILD_DIR)/generate_digits generate_digits.c $(LDFLAGS)

config: $(BUILD_DIR)/config.o $(BUILD_DIR)/tools.o
	$(CC) $(CFLAGS) -o $(BUILD_DIR)/config $(BUILD_DIR)/config.o $(BUILD_DIR)/tools.o $(LDFLAGS)

# Object file rules
$(BUILD_DIR)/xor.o: xor.c nn.h
	$(CC) $(CFLAGS) -c xor.c -o $(BUILD_DIR)/xor.o

$(BUILD_DIR)/gym.o: gym.c nn.h
	$(CC) $(CFLAGS) -c gym.c -o $(BUILD_DIR)/gym.o

$(BUILD_DIR)/accuracy.o: accuracy.c nn.h
	$(CC) $(CFLAGS) -c accuracy.c -o $(BUILD_DIR)/accuracy.o

$(BUILD_DIR)/digits.o: digits.c nn.h config.h tools.h
	$(CC) $(CFLAGS) -c digits.c -o $(BUILD_DIR)/digits.o

$(BUILD_DIR)/nn.o: nn.c nn.h tools.h
	$(CC) $(CFLAGS) -c nn.c -o $(BUILD_DIR)/nn.o

$(BUILD_DIR)/tools.o: tools.c tools.h
	$(CC) $(CFLAGS) -c tools.c -o $(BUILD_DIR)/tools.o

$(BUILD_DIR)/config.o: config.c config.h tools.h
	$(CC) $(CFLAGS) -c config.c -o $(BUILD_DIR)/config.o


# Clean rule
clean:
	rm -rf $(BUILD_DIR)
# Compiler and flags
CC = gcc
CFLAGS = -g -Wall -fopenmp -Isrc
LDFLAGS = -lm -fopenmp

# Directories
SRC_DIR = src
BUILD_DIR = build
MODELS_DIR = models

# All executables
all: directories xor gym accuracy digits generate_digits convert_optdigits

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

digits: directories $(BUILD_DIR)/digits.o $(BUILD_DIR)/nn.o $(BUILD_DIR)/tools.o $(BUILD_DIR)/config.o
	$(CC) $(CFLAGS) -o $(BUILD_DIR)/digits $(BUILD_DIR)/digits.o $(BUILD_DIR)/nn.o $(BUILD_DIR)/tools.o $(BUILD_DIR)/config.o $(LDFLAGS)

generate_digits: $(SRC_DIR)/generate_digits.c
	$(CC) $(CFLAGS) -o $(BUILD_DIR)/generate_digits $(SRC_DIR)/generate_digits.c $(LDFLAGS)

convert_optdigits: $(SRC_DIR)/convert_optdigits.c
	$(CC) $(CFLAGS) -o $(BUILD_DIR)/convert_optdigits $(SRC_DIR)/convert_optdigits.c $(LDFLAGS)

config: $(BUILD_DIR)/config.o $(BUILD_DIR)/tools.o
	$(CC) $(CFLAGS) -o $(BUILD_DIR)/config $(BUILD_DIR)/config.o $(BUILD_DIR)/tools.o $(LDFLAGS)

# Object file rules
$(BUILD_DIR)/xor.o: $(SRC_DIR)/xor.c $(SRC_DIR)/nn.h
	$(CC) $(CFLAGS) -c $(SRC_DIR)/xor.c -o $(BUILD_DIR)/xor.o

$(BUILD_DIR)/gym.o: $(SRC_DIR)/gym.c $(SRC_DIR)/nn.h
	$(CC) $(CFLAGS) -c $(SRC_DIR)/gym.c -o $(BUILD_DIR)/gym.o

$(BUILD_DIR)/accuracy.o: $(SRC_DIR)/accuracy.c $(SRC_DIR)/nn.h
	$(CC) $(CFLAGS) -c $(SRC_DIR)/accuracy.c -o $(BUILD_DIR)/accuracy.o

$(BUILD_DIR)/digits.o: $(SRC_DIR)/digits.c $(SRC_DIR)/nn.h $(SRC_DIR)/config.h $(SRC_DIR)/tools.h
	$(CC) $(CFLAGS) -c $(SRC_DIR)/digits.c -o $(BUILD_DIR)/digits.o

$(BUILD_DIR)/nn.o: $(SRC_DIR)/nn.c $(SRC_DIR)/nn.h $(SRC_DIR)/tools.h
	$(CC) $(CFLAGS) -c $(SRC_DIR)/nn.c -o $(BUILD_DIR)/nn.o

$(BUILD_DIR)/tools.o: $(SRC_DIR)/tools.c $(SRC_DIR)/tools.h
	$(CC) $(CFLAGS) -c $(SRC_DIR)/tools.c -o $(BUILD_DIR)/tools.o

$(BUILD_DIR)/config.o: $(SRC_DIR)/config.c $(SRC_DIR)/config.h $(SRC_DIR)/tools.h
	$(CC) $(CFLAGS) -c $(SRC_DIR)/config.c -o $(BUILD_DIR)/config.o


# Clean rule
clean:
	rm -rf $(BUILD_DIR)

# Compiler and flags
CC = gcc
CFLAGS = -g -Wall -fopenmp
LDFLAGS = -lm -fopenmp

BUILD_DIR = build
PROGRAMS = xor gym accuracy digits generate_digits
COMMON_OBJS = $(BUILD_DIR)/nn.o $(BUILD_DIR)/tools.o $(BUILD_DIR)/config.o

.PHONY: all clean xor gym accuracy digits generate_digits config

# All executables
all: $(addprefix $(BUILD_DIR)/,$(PROGRAMS))

# Convenience aliases (e.g. `make digits`)
xor: $(BUILD_DIR)/xor
gym: $(BUILD_DIR)/gym
accuracy: $(BUILD_DIR)/accuracy
digits: $(BUILD_DIR)/digits
generate_digits: $(BUILD_DIR)/generate_digits
config: $(BUILD_DIR)/config

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Executable rules
$(BUILD_DIR)/xor: $(BUILD_DIR)/xor.o $(COMMON_OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(BUILD_DIR)/gym: $(BUILD_DIR)/gym.o $(COMMON_OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(BUILD_DIR)/accuracy: $(BUILD_DIR)/accuracy.o $(COMMON_OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(BUILD_DIR)/digits: $(BUILD_DIR)/digits.o $(COMMON_OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(BUILD_DIR)/generate_digits: generate_digits.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD_DIR)/config: $(BUILD_DIR)/config.o $(BUILD_DIR)/tools.o | $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Object file rules
$(BUILD_DIR)/xor.o: xor.c nn.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/gym.o: gym.c nn.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/accuracy.o: accuracy.c nn.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/digits.o: digits.c nn.h config.h tools.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/nn.o: nn.c nn.h tools.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/tools.o: tools.c tools.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/config.o: config.c config.h tools.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Clean rule
clean:
	rm -rf $(BUILD_DIR) xor gym accuracy digits generate_digits config *.o *.model

CC ?= gcc
CPPFLAGS := -Isrc
CFLAGS ?= -O2 -g
CFLAGS += -std=c11 -Wall -Wextra -Wpedantic
LDLIBS := -lm

BUILD_DIR := build
MODEL_DIR := models
COMMON_OBJECTS := $(BUILD_DIR)/nn.o $(BUILD_DIR)/dataset.o $(BUILD_DIR)/config.o
PROGRAMS := digits accuracy gym convert_optdigits

.PHONY: all check clean $(PROGRAMS)

all: $(addprefix $(BUILD_DIR)/,$(PROGRAMS)) | $(MODEL_DIR)

$(PROGRAMS): %: $(BUILD_DIR)/%

$(BUILD_DIR) $(MODEL_DIR):
	mkdir -p $@

$(BUILD_DIR)/digits: $(BUILD_DIR)/digits.o $(COMMON_OBJECTS) | $(BUILD_DIR) $(MODEL_DIR)
	$(CC) $(CFLAGS) $^ $(LDLIBS) -o $@

$(BUILD_DIR)/accuracy: $(BUILD_DIR)/accuracy.o $(COMMON_OBJECTS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $^ $(LDLIBS) -o $@

$(BUILD_DIR)/gym: $(BUILD_DIR)/gym.o $(COMMON_OBJECTS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $^ $(LDLIBS) -o $@

$(BUILD_DIR)/convert_optdigits: $(BUILD_DIR)/convert_optdigits.o | $(BUILD_DIR)
	$(CC) $(CFLAGS) $^ $(LDLIBS) -o $@

$(BUILD_DIR)/test_core: $(BUILD_DIR)/test_core.o $(COMMON_OBJECTS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $^ $(LDLIBS) -o $@

$(BUILD_DIR)/%.o: src/%.c | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR)/test_core.o: tests/test_core.c | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -MMD -MP -c $< -o $@

check: all $(BUILD_DIR)/test_core
	./$(BUILD_DIR)/test_core
	./$(BUILD_DIR)/convert_optdigits datasets/optdigits.tra $(BUILD_DIR)/train.csv datasets/optdigits.tes $(BUILD_DIR)/test.csv
	cmp datasets/UCI_digits_train.csv $(BUILD_DIR)/train.csv
	cmp datasets/UCI_digits_test.csv $(BUILD_DIR)/test.csv

clean:
	rm -rf $(BUILD_DIR)

-include $(wildcard $(BUILD_DIR)/*.d)

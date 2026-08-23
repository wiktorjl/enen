CC ?= gcc
EMCC ?= emcc
NODE ?= node
CPPFLAGS := -Isrc
CFLAGS ?= -O2 -g
CFLAGS += -std=c11 -Wall -Wextra -Wpedantic
LDLIBS := -lm

BUILD_DIR := build
MODEL_DIR := models
COMMON_OBJECTS := $(BUILD_DIR)/nn.o $(BUILD_DIR)/dataset.o $(BUILD_DIR)/config.o
WEB_API_OBJECTS := $(BUILD_DIR)/web_api.o $(BUILD_DIR)/nn.o $(BUILD_DIR)/dataset.o
PROGRAMS := digits accuracy gym convert_optdigits
WEB_MODULE := webapp/enen.js
WEB_WASM := webapp/enen.wasm
WEB_SOURCE_ASSET := webapp/assets/nn.c
WEB_EXPORTS := '["_web_initialize","_web_reset_model","_web_configure_model","_web_save_model","_web_load_model","_web_train_batch","_web_train_epoch","_web_evaluate","_web_accuracy","_web_loss","_web_training_samples","_web_test_samples","_web_epochs_trained","_web_epoch_position","_web_synthetic_samples","_web_num_layers","_web_layer_size","_web_activation","_web_layer_weights","_web_activation_snapshot","_web_activation_count","_web_last_training_label","_web_activation_version","_web_input_buffer","_web_inspect_input","_web_predict","_web_probability","_web_clear_synthetic_samples","_web_add_synthetic_sample","_web_copy_test_sample","_web_copy_training_sample","_web_test_label","_web_cleanup"]'

.PHONY: all check web web-check browser-check clean $(PROGRAMS)

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

$(BUILD_DIR)/test_web_api: $(BUILD_DIR)/test_web_api.o $(WEB_API_OBJECTS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $^ $(LDLIBS) -o $@

web: $(WEB_MODULE) $(WEB_WASM) $(WEB_SOURCE_ASSET)

web-check:
	$(MAKE) --always-make web
	$(NODE) --check webapp/app.js
	$(NODE) --check $(WEB_MODULE)
	$(NODE) tests/test_wasm.mjs
	! grep -Eq 'class NeuralNetwork|trainOne|fromCModel' webapp/app.js
	cmp src/nn.c $(WEB_SOURCE_ASSET)
	cmp datasets/optdigits.tra webapp/assets/optdigits.tra
	cmp datasets/optdigits.tes webapp/assets/optdigits.tes

browser-check:
	$(NODE) --check webapp/app.js
	$(NODE) tests/test_browser.mjs

$(WEB_MODULE) $(WEB_WASM) &: src/web_api.c src/web_api.h src/nn.c src/nn.h src/dataset.c src/dataset.h src/config.h Makefile
	$(EMCC) $(CPPFLAGS) -O3 -std=c11 src/web_api.c src/nn.c src/dataset.c \
		--no-entry -sWASM=1 -sMODULARIZE=1 -sEXPORT_NAME=createEnenModule \
		-sENVIRONMENT=web -sFILESYSTEM=1 -sALLOW_MEMORY_GROWTH=1 \
		-sEXPORTED_FUNCTIONS=$(WEB_EXPORTS) \
		-sEXPORTED_RUNTIME_METHODS='["FS","ccall"]' -o $(WEB_MODULE)
	chmod 0644 $(WEB_MODULE) $(WEB_WASM)

$(WEB_SOURCE_ASSET): src/nn.c
	cp $< $@
	chmod 0644 $@

$(BUILD_DIR)/%.o: src/%.c | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR)/test_core.o: tests/test_core.c | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR)/test_web_api.o: tests/test_web_api.c | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -MMD -MP -c $< -o $@

check: all $(BUILD_DIR)/test_core $(BUILD_DIR)/test_web_api
	./$(BUILD_DIR)/test_core
	./$(BUILD_DIR)/test_web_api
	./$(BUILD_DIR)/convert_optdigits datasets/optdigits.tra $(BUILD_DIR)/train.csv datasets/optdigits.tes $(BUILD_DIR)/test.csv
	cmp datasets/UCI_digits_train.csv $(BUILD_DIR)/train.csv
	cmp datasets/UCI_digits_test.csv $(BUILD_DIR)/test.csv

clean:
	rm -rf $(BUILD_DIR) $(WEB_MODULE) $(WEB_WASM) $(WEB_SOURCE_ASSET)

-include $(wildcard $(BUILD_DIR)/*.d)

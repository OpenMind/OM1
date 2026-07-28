GO := go
GOBIN := $(shell $(GO) env GOPATH)/bin
BINARY_NAME := om1
GO_MODULE := github.com/openmind/om1
CMD_DIR := ./cmd
BUILD_DIR := ./build
GO_FILES := $(shell find . -name '*.go' -type f)

LDFLAGS := -s -w
BUILD_FLAGS := -v

CONFIG ?= conversation

ZENOH_C_VERSION=1.9.0
ZENOH_C_DIR=.zenoh-c
ZENOH_C_ABS_DIR=$(shell pwd)/$(ZENOH_C_DIR)
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

ifeq ($(UNAME_S),Linux)
	ifeq ($(UNAME_M),x86_64)
		ZENOH_PLATFORM=x86_64-unknown-linux-gnu
	else ifeq ($(UNAME_M),aarch64)
		ZENOH_PLATFORM=aarch64-unknown-linux-gnu
	endif
	DYLD_VAR=LD_LIBRARY_PATH
else ifeq ($(UNAME_S),Darwin)
	ifeq ($(UNAME_M),arm64)
		ZENOH_PLATFORM=aarch64-apple-darwin
	else
		ZENOH_PLATFORM=x86_64-apple-darwin
	endif
	DYLD_VAR=DYLD_LIBRARY_PATH
endif

ZENOH_URL=https://github.com/eclipse-zenoh/zenoh-c/releases/download/$(ZENOH_C_VERSION)/zenoh-c-$(ZENOH_C_VERSION)-$(ZENOH_PLATFORM)-standalone.zip

# onnxruntime and the Silero VAD model are only needed for the optional
# enable_vad_latency ASR feature. onnxruntime_go dlopens the shared library
# at *runtime* (not link time), so unlike zenoh-c, neither download is a
# prerequisite of build/test -- run them explicitly to use the feature.
ONNXRUNTIME_VERSION=1.27.1
ONNXRUNTIME_DIR=.onnxruntime
ifeq ($(UNAME_S),Linux)
	ifeq ($(UNAME_M),x86_64)
		ONNXRUNTIME_PLATFORM=linux-x64
	else ifeq ($(UNAME_M),aarch64)
		ONNXRUNTIME_PLATFORM=linux-aarch64
	endif
else ifeq ($(UNAME_S),Darwin)
	ifeq ($(UNAME_M),arm64)
		ONNXRUNTIME_PLATFORM=osx-arm64
	else
		ONNXRUNTIME_PLATFORM=osx-x86_64
	endif
endif
ONNXRUNTIME_ARCHIVE=onnxruntime-$(ONNXRUNTIME_PLATFORM)-$(ONNXRUNTIME_VERSION)
ONNXRUNTIME_URL=https://github.com/microsoft/onnxruntime/releases/download/v$(ONNXRUNTIME_VERSION)/$(ONNXRUNTIME_ARCHIVE).tgz

VAD_MODEL_DIR=models
VAD_MODEL_PATH=$(VAD_MODEL_DIR)/silero_vad_v5.onnx
VAD_MODEL_URL=https://huggingface.co/runanywhere/silero-vad-v5/resolve/main/silero_vad.onnx

export CGO_ENABLED=1
export CGO_CFLAGS=-I$(ZENOH_C_ABS_DIR)/include
export CGO_LDFLAGS=-L$(ZENOH_C_ABS_DIR)/lib -lzenohc -Wl,-rpath,$(ZENOH_C_ABS_DIR)/lib

.PHONY: all
all: lint build

.PHONY: help
help:
	@echo "Available targets:"
	@echo "  all              - Run lint and build"
	@echo "  build            - Build the binary"
	@echo "  run              - Build and run with config (CONFIG=greeting by default)"
	@echo "  dev              - Run without building (go run)"
	@echo "  lint             - Run linter (golangci-lint)"
	@echo "  fmt              - Format Go code"
	@echo "  vet              - Run go vet"
	@echo "  test             - Run tests with coverage"
	@echo "  test-integration - Run end-to-end integration tests"
	@echo "  test-coverage    - Generate HTML coverage report"
	@echo "  clean            - Clean build artifacts"
	@echo "  deps             - Download and tidy dependencies"
	@echo "  deps-update      - Update dependencies"
	@echo "  install          - Install binary to GOPATH/bin"
	@echo "  check            - Run fmt, vet, lint, and test"
	@echo "  download-zenohc      - Download and extract zenohc library"
	@echo "  download-onnxruntime - Download onnxruntime (needed for enable_vad_latency)"
	@echo "  download-vad-model   - Download the Silero VAD v5 model (needed for enable_vad_latency)"
	@echo "  list-configs         - List available configuration files"

download-zenohc:
	@echo "Downloading zenoh-c $(ZENOH_C_VERSION) for $(ZENOH_PLATFORM)..."
	@mkdir -p $(ZENOH_C_DIR)
	@if [ ! -f "$(ZENOH_C_DIR)/lib/libzenohc.dylib" ] && [ ! -f "$(ZENOH_C_DIR)/lib/libzenohc.so" ]; then \
		echo "Fetching $(ZENOH_URL)..."; \
		curl -sSL -o /tmp/zenoh-c.zip $(ZENOH_URL); \
		unzip -q /tmp/zenoh-c.zip -d $(ZENOH_C_DIR); \
		rm /tmp/zenoh-c.zip; \
		echo "zenoh-c installed to $(ZENOH_C_DIR)"; \
		if [ "$(UNAME_S)" = "Darwin" ]; then \
			echo "Patching dylib install names..."; \
			if [ -f "$(ZENOH_C_ABS_DIR)/lib/libzenohc.dylib" ]; then \
				install_name_tool -id "@rpath/libzenohc.dylib" "$(ZENOH_C_ABS_DIR)/lib/libzenohc.dylib"; \
			fi; \
		fi; \
	else \
		echo "zenoh-c already installed in $(ZENOH_C_DIR)"; \
	fi

.PHONY: download-onnxruntime
download-onnxruntime:
	@echo "Downloading onnxruntime $(ONNXRUNTIME_VERSION) for $(ONNXRUNTIME_PLATFORM)..."
	@mkdir -p $(ONNXRUNTIME_DIR)
	@if [ ! -f "$(ONNXRUNTIME_DIR)/lib/libonnxruntime.so" ] && [ ! -f "$(ONNXRUNTIME_DIR)/lib/libonnxruntime.dylib" ]; then \
		echo "Fetching $(ONNXRUNTIME_URL)..."; \
		curl -sSL -o /tmp/onnxruntime.tgz $(ONNXRUNTIME_URL); \
		tar xzf /tmp/onnxruntime.tgz --strip-components=1 -C $(ONNXRUNTIME_DIR); \
		rm /tmp/onnxruntime.tgz; \
		echo "onnxruntime installed to $(ONNXRUNTIME_DIR)"; \
	else \
		echo "onnxruntime already installed in $(ONNXRUNTIME_DIR)"; \
	fi

.PHONY: download-vad-model
download-vad-model:
	@echo "Downloading Silero VAD v5 model..."
	@mkdir -p $(VAD_MODEL_DIR)
	@if [ ! -f "$(VAD_MODEL_PATH)" ]; then \
		echo "Fetching $(VAD_MODEL_URL)..."; \
		curl -sSL -o $(VAD_MODEL_PATH) $(VAD_MODEL_URL); \
		echo "VAD model installed to $(VAD_MODEL_PATH)"; \
	else \
		echo "VAD model already installed at $(VAD_MODEL_PATH)"; \
	fi

.PHONY: build
build: download-zenohc
	@echo "Building $(BINARY_NAME)..."
	@mkdir -p $(BUILD_DIR)
	$(DYLD_VAR)=$(ZENOH_C_ABS_DIR)/lib $(GO) build $(BUILD_FLAGS) -ldflags "$(LDFLAGS)" -o $(BUILD_DIR)/$(BINARY_NAME) $(CMD_DIR)
	@echo "Binary built: $(BUILD_DIR)/$(BINARY_NAME)"

.PHONY: run
run: build
	@echo "Running $(BINARY_NAME) with config: $(CONFIG)"
	$(BUILD_DIR)/$(BINARY_NAME) -config $(CONFIG)

.PHONY: dev
dev: download-zenohc
	@echo "Running in dev mode with config: $(CONFIG)"
	$(DYLD_VAR)=$(ZENOH_C_ABS_DIR)/lib $(GO) run $(CMD_DIR) -config $(CONFIG) -log-level debug

.PHONY: lint
lint: download-zenohc
	@echo "Running linter..."
	@if ! command -v golangci-lint > /dev/null; then \
		echo "golangci-lint not found. Installing..."; \
		$(DYLD_VAR)=$(ZENOH_C_ABS_DIR)/lib $(GO) install github.com/golangci/golangci-lint/cmd/golangci-lint@latest; \
	fi
	$(DYLD_VAR)=$(ZENOH_C_ABS_DIR)/lib golangci-lint run ./...

.PHONY: fmt
fmt: download-zenohc
	@echo "Formatting Go code..."
	$(DYLD_VAR)=$(ZENOH_C_ABS_DIR)/lib $(GO) fmt ./...

.PHONY: vet
vet: download-zenohc
	@echo "Running go vet..."
	$(DYLD_VAR)=$(ZENOH_C_ABS_DIR)/lib $(GO) vet ./...

.PHONY: test
test: download-zenohc
	@echo "Running tests..."
	$(DYLD_VAR)=$(ZENOH_C_ABS_DIR)/lib $(GO) test -v -race -coverprofile=coverage.out ./...

.PHONY: test-integration
test-integration: download-zenohc
	@echo "Running integration tests..."
	$(DYLD_VAR)=$(ZENOH_C_ABS_DIR)/lib $(GO) test -v -tags=integration ./test/integration/...

.PHONY: test-coverage
test-coverage: test
	@echo "Generating coverage report..."
	$(DYLD_VAR)=$(ZENOH_C_ABS_DIR)/lib $(GO) tool cover -html=coverage.out -o coverage.html
	@echo "Coverage report: coverage.html"

.PHONY: clean
clean:
	@echo "Cleaning..."
	@rm -rf $(BUILD_DIR)
	@rm -f coverage.out coverage.html
	@echo "Clean complete"

.PHONY: deps
deps: download-zenohc
	@echo "Downloading dependencies..."
	$(DYLD_VAR)=$(ZENOH_C_ABS_DIR)/lib $(GO) mod download
	$(DYLD_VAR)=$(ZENOH_C_ABS_DIR)/lib $(GO) mod tidy

.PHONY: deps-update
deps-update: download-zenohc
	@echo "Updating dependencies..."
	$(DYLD_VAR)=$(ZENOH_C_ABS_DIR)/lib $(GO) get -u ./...
	$(DYLD_VAR)=$(ZENOH_C_ABS_DIR)/lib $(GO) mod tidy

.PHONY: install
install: build
	@echo "Installing $(BINARY_NAME) to $(GOBIN)..."
	@cp $(BUILD_DIR)/$(BINARY_NAME) $(GOBIN)/
	@echo "Installed to $(GOBIN)/$(BINARY_NAME)"

.PHONY: check
check: fmt vet lint test

.PHONY: list-configs
list-configs:
	@echo "Available configurations:"
	@ls -1 ./config/*.json5 | xargs -n 1 basename | sed 's/\.json5//'

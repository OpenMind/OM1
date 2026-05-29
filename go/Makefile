GO := go
GOBIN := $(shell $(GO) env GOPATH)/bin
BINARY_NAME := om1
GO_MODULE := github.com/openmind/om1
CMD_DIR := ./cmd
BUILD_DIR := ../build
GO_FILES := $(shell find . -name '*.go' -type f)

LDFLAGS := -s -w
BUILD_FLAGS := -v

CONFIG ?= greeting

.PHONY: all
all: lint build

.PHONY: help
help:
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

.PHONY: build
build:
	@echo "Building $(BINARY_NAME)..."
	@mkdir -p $(BUILD_DIR)
	$(GO) build $(BUILD_FLAGS) -ldflags "$(LDFLAGS)" -o $(BUILD_DIR)/$(BINARY_NAME) $(CMD_DIR)
	@echo "Binary built: $(BUILD_DIR)/$(BINARY_NAME)"

.PHONY: run
run: build
	@echo "Running $(BINARY_NAME) with config: $(CONFIG)"
	$(BUILD_DIR)/$(BINARY_NAME) -config ../config/$(CONFIG).json5

.PHONY: dev
dev: ## Run without building (go run)
	@echo "Running in dev mode with config: $(CONFIG)"
	$(GO) run $(CMD_DIR) -config ../config/$(CONFIG).json5

.PHONY: lint
lint: ## Run linter (golangci-lint)
	@echo "Running linter..."
	@if ! command -v golangci-lint > /dev/null; then \
		echo "golangci-lint not found. Installing..."; \
		$(GO) install github.com/golangci/golangci-lint/cmd/golangci-lint@latest; \
	fi
	golangci-lint run ./...

.PHONY: fmt
fmt:
	@echo "Formatting Go code..."
	$(GO) fmt ./...

.PHONY: vet
vet:
	@echo "Running go vet..."
	$(GO) vet ./...

.PHONY: test
test:
	@echo "Running tests..."
	$(GO) test -v -race -coverprofile=coverage.out ./...

.PHONY: test-coverage
test-coverage: test
	@echo "Generating coverage report..."
	$(GO) tool cover -html=coverage.out -o coverage.html
	@echo "Coverage report: coverage.html"

.PHONY: clean
clean:
	@echo "Cleaning..."
	@rm -rf $(BUILD_DIR)
	@rm -f coverage.out coverage.html
	@echo "Clean complete"

.PHONY: deps
deps:
	@echo "Downloading dependencies..."
	$(GO) mod download
	$(GO) mod tidy

.PHONY: deps-update
deps-update:
	@echo "Updating dependencies..."
	$(GO) get -u ./...
	$(GO) mod tidy

.PHONY: instal
	@echo "Installing $(BINARY_NAME) to $(GOBIN)..."
	@cp $(BUILD_DIR)/$(BINARY_NAME) $(GOBIN)/
	@echo "Installed to $(GOBIN)/$(BINARY_NAME)"

.PHONY: check
check: fmt vet lint test

.PHONY: list-configs
list-configs:
	@echo "Available configurations:"
	@ls -1 ../config/*.json5 | xargs -n 1 basename | sed 's/\.json5//'

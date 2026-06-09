# Copilot Instructions

## Tooling / Dependencies
- We use **Make** for building, testing, and running the project.
- Dependencies are managed via Go modules (`go.mod` and `go.sum`).
- Use these make commands:
  - Build: `make build`
  - Test: `make test`
  - Test with coverage: `make test-coverage`
  - Run: `make run CONFIG=<config_name>`
  - Dev mode: `make dev CONFIG=<config_name>`
  - Lint: `make lint`
  - Format: `make fmt`
  - Check all: `make check` (runs fmt, vet, lint, and test)
  - Clean: `make clean`
- When adding a new dependency, use `go get <package>` and run `make deps` to tidy.
- Keep the dependency list minimal and prefer well-maintained packages.

## Coding Style
- Use Go 1.21+
- Follow Go conventions and idiomatic patterns
- Use proper error handling (never ignore errors)
- Follow the standard Go project layout
- Use `gofmt` for formatting (enforced by `make fmt`)

## Architecture
- Use modular design
- Separate concerns: inputs, actions, backgrounds, LLM providers
- Keep business logic out of HTTP handlers
- Use interfaces for abstractions

## Testing
- Use Go's built-in testing framework
- Add unit tests for all new functions
- Aim for 90% code coverage
- Run tests with race detector: `make test`
- Use table-driven tests where appropriate
- Mock external dependencies using interfaces

## Comments
- Write clear package documentation
- Document exported functions and types
- Avoid obvious comments
- Use godoc conventions

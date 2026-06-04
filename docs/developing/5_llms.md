---
title: LLMs
description: "LLM Integration"
icon: brain-circuit
---

OM1's LLM integration is intended to make it easy to (1) send `input` information to LLMs and then (2) route LLM responses to various system actions, such as `speak` and `move`. The OM1 system integrates various concrete implementations of Large Language Models (LLMs), each designed to address different requirements and interaction patterns. These implementations manage API communication, conversation history, and the processing of structured responses, particularly for function calls that trigger agent actions. The framework ensures a consistent interface, allowing the system to interchangeably utilize diverse LLM backends.

OM1 also supports per-mode LLM configuration. If a mode specifies its own LLM, it takes precedence over the top-level cortex_llm setting. This allows different modes to use different models based on their specific requirements.

The plugins handle authentication, API communication, prompt formatting, response parsing, and conversation history management. LLM plugin examples are located in `plugins/llm`: [**Code**](https://github.com/OpenMind/OM1/tree/main/plugins/llm).

## Endpoint Overview

```bash
# Base URL: https://api.openmind.com/

POST /api/core/{provider}/chat/completions    # Single agent
DELETE /api/core/agent/memory                 # Multi agent memory wipe
```

## LLM Modes

OM1 supports three LLM execution strategies depending on your latency, quality, and reliability requirements.

| Mode | Description | Performance |
|------|-------------|----------|
| **Single** | One LLM processes all requests | Good — fast, but limited capability |
| **Dual** | Local + cloud LLMs in parallel | Better — higher accuracy, but slower |
| **Parallel** | N specialized LLMs run simultaneously | Best — fastest and most capable |

### Single LLM Integration

For testing and introductory educational purposes, we integrate with multiple language models (LLMs) to provide chat completion via a `POST /api/core/{provider}/chat/completions` endpoint. Each LLM plugin takes fused input data (the `prompt`) and sends it to an LLM. The response is then parsed and provided to `internal/runtime/cortex.go` for distribution to the system actions:

```go
response, err := client.ChatCompletions(ctx, &ChatRequest{
    Model:    config.Model,
    Messages: messages,
    ResponseFormat: outputModel,
    Timeout:  config.Timeout,
})

parsedResponse := outputModel.Validate(response.Choices[0].Message.Content)
return parsedResponse
```

The standard output model is defined in `internal/llm/output_model.go`.

Example config:

```json5
  "cortex_llm": {
    "type": "OpenAILLM",     // The class name of the LLM plugin you wish to use
    "config": {
      "model": "model_name", // Optional: If you want to switch to a specific model. Refer the list of supported models below
      "base_url": "",        // Optional: URL of the LLM endpoint
      "agent_name": "Iris",  // Optional: Name of the agent
      "history_length": 10   // The number of input->action cycles to provide to the LLM as historical context
    }
  }
```

### Dual LLM support

OM1 implements a dual-LLM response mechanism that combines both local and cloud-based models to optimize response quality and latency.

- Local model: Qwen3-30B (on-device)
- Cloud model: GPT-4.1

Example config:

```json5
  "cortex_llm": {
    "type": "DualLLM",      // The class name of the LLM plugin you wish to use
    "config": {
        "local_llm_type": "QwenLLM",                // The class name of the LLM plugin you wish to use for local llm
        "local_llm_config": {"model": "RedHatAI/Qwen3-30B-A3B-quantized.w4a16"},        // model name you wish to use
        "cloud_llm_type": "OpenAILLM",              // The class name of the LLM plugin you wish to use for cloud llm
        "cloud_llm_config": {"model": "gpt-4.1"}    // model name you wish to use
    }
}
```

**How It Works**

1. For each request, OM1 sends the prompt to both the local and cloud LLMs in parallel.

2. The system waits up to 3.2 seconds for responses.

3. If both models return a response within the threshold:

    - The two responses are evaluated by the local LLM.

    - The local LLM selects the better response as the final output.

4. If only one model responds within the threshold:

    That response is used directly as the final output.

This approach ensures fast responses while leveraging cloud models for higher-quality outputs when available.

### Parallel LLM

Multiple LLMs run in parallel, each handling specific actions they are capable of. Results stream as they complete, allowing the cortex to execute actions immediately without waiting for all LLMs.

Example config:

```json5
  "cortex_llm": {
    "type": "ParallelLLM",      // The class name of the LLM plugin you wish to use
    "config": {
        "llms": [
            {
                "llm_type": "OpenAILLM",                // The class name of the LLM plugin you wish to use
                "llm_config": {"model": "gpt-4.1"},     // model name you wish to use
                "action_filter": ["speak", "emotion"]   // preferred action for the model
            },
            {
                "llm_type": "QwenLLM",                       // The class name of the LLM plugin you wish to use
                "llm_config": {"model": "RedHatAI/Qwen3-30B-A3B-quantized.w4a16"},      // model name you wish to use
                "action_filter": ["move", "navigate"]       // preferred action for the model
            },
            {
                "llm_type": "DeepSeekLLM",                   // The class name of the LLM plugin you wish to use
                "llm_config": {"model": "deepseek-chat"},    // preferred action for the model
                "action_filter": ["search", "analyze"]       // preferred action for the model
            }
        ],
        "execute_immediately": true
    }
}
```

## Local LLMs

The system supports on-device inference using the Qwen3-30B local LLM. This enables low-latency responses and allows certain workloads to run entirely on the device without relying on cloud connectivity.

### Ollama Integration

[Ollama](https://ollama.ai) provides an easy way to run open-source models locally. OM1 supports Ollama through the `OllamaLLM` plugin.

**Prerequisites:**
1. Install Ollama: https://ollama.ai
2. Pull a model: `ollama pull llama3.2`
3. Ensure Ollama is running: `ollama serve`

**Configuration:**
```json
"cortex_llm": {
  "type": "OllamaLLM",
  "config": {
    "model": "llama3.2",
    "base_url": "http://localhost:11434",
    "temperature": 0.7,
    "num_ctx": 4096,
    "timeout": 120
  }
}
```

**Run with Ollama:**
```bash
make run CONFIG=ollama
```

### Agent Architecture

The system employs four primary agents that work together:

- **Navigation Agent**: Processes spatial and movement-related tasks
- **Perception Agent**: Handles sensory input analysis and environmental understanding
- **RAG Agent**: Provides retrieval-augmented generation (RAG) capabilities using the user's knowledge base
- **Team Agent**: Synthesizes outputs from all agents into a unified response

### Main API Endpoint

```go
endpoint := "/api/core/{provider}/chat/completions"

headers := map[string]string{
    "Authorization": "Bearer " + config.APIKey,
    "Content-Type":  "application/json",
}

request := ChatRequest{
    SystemPrompt:      ioProvider.FuserSystemPrompt,
    Inputs:            ioProvider.FuserInputs,
    Model:             config.Model,
    ResponseFormat:    outputModel.JSONSchema(),
    StructuredOutputs: true,
}

response, err := httpClient.Post(endpoint, request, headers)
output := response.Content
return outputModel.Validate(output)
```

### Supported Models

```go
var OpenAISupportedModels = []string{"gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-5", "gpt-5-mini", "gpt-5-nano"}
```

```go
var DeepSeekSupportedModels = []string{"deepseek-chat"}
```

```go
var GeminiSupportedModels = []string{"gemini-3.5-flash", "gemini-3.1-pro-preview", "gemini-3.1-flash-lite", "gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro"}
```

```go
var XAISupportedModels = []string{"grok-2-latest", "grok-3-beta", "grok-4-latest", "grok-4"}
```

```go
var NearAISupportedModels = []string{"qwen3-30b-a3b-instruct-2507", "qwen2.5-vl-72b-instruct", "qwen-2.5-7b-instruct"}
```

```go
var OpenRouterSupportedModels = []string{"meta-llama/llama-3.1-70b-instruct", "meta-llama/llama-3.3-70b-instruct", "anthropic/claude-sonnet-4.5", "anthropic/claude-opus-4.1"}
```

```go
// Ollama supports any model from https://ollama.ai/library
var OllamaSupportedModels = []string{"llama3.2", "llama3.1", "mistral", "phi3", "gemma2", "qwen2.5", "codellama", "llava"}
```

```go
// Local LLM
var LocalLLMModels = []string{"Qwen3-30B"}
```

## Examples

### A Smart Dog

Imagine you would like to program a smart dog. Describe the desired capabilities and behaviors of the dog in `system_prompt_base`. For example:

```json5
"system_prompt_base": "You are an intelligent robotic dog companion designed to be helpful, loyal, and engaging. Your primary goals are to: (1) Provide companionship through interactive play and conversation, (2) Assist with basic household tasks and monitoring, (3) Learn and adapt to your owner's preferences and routines, and (4) Maintain a playful yet responsible demeanor. You can move around, speak clearly, express emotions through body language, and respond to voice commands. Always prioritize safety and be eager to please while maintaining your dog-like personality traits of curiosity, loyalty, and enthusiasm."
```

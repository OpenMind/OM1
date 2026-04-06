---
title: LLMs
description: "LLM Integration"
icon: brain-circuit
---

OM1's LLM integration is intended to make it easy to (1) send `input` information to LLMs and then (2) route LLM responses to various system actions, such as `speak` and `move`. The OM1 system integrates various concrete implementations of Large Language Models (LLMs), each designed to address different requirements and interaction patterns. These implementations manage API communication, conversation history, and the processing of structured responses, particularly for function calls that trigger agent actions. The framework ensures a consistent interface, allowing the system to interchangeably utilize diverse LLM backends.

OM1 supports per-mode LLM configuration. If a mode specifies its own LLM, it takes precedence over the top-level cortex_llm setting. This allows different modes to use different models based on their specific requirements.

The plugins handle authentication, API communication, prompt formatting, response parsing, and conversation history management. LLM plugin examples are located in `src/llm/plugins`: [**Code**](https://github.com/OpenMind/OM1/tree/main/src/llm/plugins).

## Endpoint Overview

```bash
# Base URL: https://api.openmind.com/

POST /api/core/{provider}/chat/completions    # Single agent
DELETE /api/core/agent/memory                 # Multi agent memory wipe
```

### Single-Agent LLM Integration

For testing and introductory educational purposes, we integrate with multiple language models (LLMs) to provide chat completion via a `POST /api/core/{provider}/chat/completions` endpoint. Each LLM plugin takes fused input data (the `prompt`) and sends it to an LLM. The response is then parsed and provided to `runtime/cortex.py` for distribution to the system actions:

```python
response = await self._client.beta.chat.completions.parse(
    model=self._config.model,
    messages=[*messages, {"role": "user", "content": prompt}],
    response_format=self._output_model,
    timeout=self._config.timeout,
)

message_content = response.choices[0].message.content
parsed_response = self._output_model.model_validate_json(message_content)

return parsed_response
```

The standard `pydantic` output model is defined in `src/llm/output_model.py`.

```bash
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

### How It Works

1. For each request, OM1 sends the prompt to both the local and cloud LLMs in parallel.

2. The system waits up to 3.2 seconds for responses.

3. If both models return a response within the threshold:

    - The two responses are evaluated by the local LLM.

    - The local LLM selects the better response as the final output.

4. If only one model responds within the threshold:

    That response is used directly as the final output.

This approach ensures fast responses while leveraging cloud models for higher-quality outputs when available.

### Parallel LLM

A system where N LLMs (any number: 1, 2, 3, or more) run in parallel,
each generating specific actions they are capable of handling.
Results are yielded as each LLM completes, allowing the cortex
to execute actions immediately without waiting for all LLMs.

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
uv run src/run.py ollama
```

### Agent Architecture

The system employs four primary agents that work together:

- **Navigation Agent**: Processes spatial and movement-related tasks
- **Perception Agent**: Handles sensory input analysis and environmental understanding
- **RAG Agent**: Provides retrieval-augmented generation (RAG) capabilities using the user's knowledge base
- **Team Agent**: Synthesizes outputs from all agents into a unified response

### Main API Endpoint

```python
    self.endpoint = "https://api.openmind.com/api/core/agent"

    headers = {
        "Authorization": f"Bearer {self._config.api_key}",
        "Content-Type": "application/json",
    }

    request = {
        "system_prompt": self.io_provider.fuser_system_prompt,
        "inputs": self.io_provider.fuser_inputs,
        "model": self._config.model,
        "response_format": self._output_model.model_json_schema(),
        "structured_outputs": True,
    }

    logging.debug(f"MultiLLM system_prompt: {request['system_prompt']}")
    logging.debug(f"MultiLLM inputs: {request['inputs']}")
    logging.debug(f"MultiLLM available_actions: {request['available_actions']}")

    response = requests.post(
        self.endpoint,
        json=request,
        headers=headers,
    )

    output = response.json().get("content")
    return self._output_model.model_validate_json(output)
```

### API Debug Response Structure

In addition to the response flowing to OM1, which contains actions the robot should perform, there is an additional response you can use for debugging and to observe token usage ("usage").

```json
{
    "content": "Synthesized response from team agent",
    "model": "gpt-4.1-nano",
    "agent_contents": {
        "team_agent": "Team agent output",
        "navigation_agent": "Navigation agent output",
        "perception_agent": "Perception agent output",
        "rag_agent": "RAG agent context"
    },
    "conversation_id": "unique-conversation-id",
    "usage": {
        "team_agent": {},
        "navigation_agent": {},
        "perception_agent": {},
        "rag_agent": {}
    },
    "duration": {
        "team_agent": 0.5,
        "navigation_agent": 0.3,
        "perception_agent": 0.4,
        "rag_agent": 0.2
    },
    "total_duration": 1.4
}
```

### Supported Models

```python
OPENAI_SUPPORTED_MODELS = ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-5", "gpt-5-mini", "gpt-5-nano"]
```

```python
DEEPSEEK_SUPPORTED_MODELS = ["deepseek-chat"]
```

```python
GEMINI_SUPPORTED_MODELS = ["gemini-3.1-pro-preview", "gemini-3.1-flash-lite-preview", "gemini-3-pro-preview", "gemini-3-flash-preview", "gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro"]
```

```python
X_AI_SUPPORTED_MODELS = ["grok-2-latest", "grok-3-beta", "grok-4-latest", "grok-4"]
```

```python
NEAR_AI_SUPPORTED_MODELS = ["qwen3-30b-a3b-instruct-2507", "qwen2.5-vl-72b-instruct", "qwen-2.5-7b-instruct"]
```

```python
OPENROUTER_SUPPORTED_MODELS = ["meta-llama/llama-3.1-70b-instruct", "meta-llama/llama-3.3-70b-instruct", "anthropic/claude-sonnet-4.5", "anthropic/claude-opus-4.1"]
```

```python
# Ollama supports any model from https://ollama.ai/library
OLLAMA_SUPPORTED_MODELS = ["llama3.2", "llama3.1", "mistral", "phi3", "gemma2", "qwen2.5", "codellama", "llava"]
```

```python
Local LLM = ["Qwen3-30B"]
```

### Memory Management

The system includes memory capabilities at `/api/core/agent/memory`:

```bash
DELETE /api/core/agent/memory
```

- Session-based memory storage via API keys
- Graph memory integration using Zep
- Conversation history tracking

### RAG Integration (Currently Disabled)

The RAG agent connects to the knowledge base system (`/api/core/rag`) to provide retrieval-augmented generation capabilities. To use RAG with your documents:

1. **Upload Documents**: Visit [https://portal.openmind.com/machines](https://portal.openmind.com/machines) to upload your documents and files to your knowledge base
2. **Ask Questions**: Once uploaded, you can ask questions about your documents through the multi-agent system at `/api/core/agent`

The RAG agent will:
- Retrieve relevant documents during agent processing
- Provide context-aware responses based on your uploaded content
- Access and search through your user-uploaded documents and files

## Examples

### A Smart Dog

Imagine you would like to program a smart dog. Describe the desired capabilities and behaviors of the dog in `system_prompt_base`. For example:

```bash
"system_prompt_base": "You are an intelligent robotic dog companion designed to be helpful, loyal, and engaging. Your primary goals are to: (1) Provide companionship through interactive play and conversation, (2) Assist with basic household tasks and monitoring, (3) Learn and adapt to your owner's preferences and routines, and (4) Maintain a playful yet responsible demeanor. You can move around, speak clearly, express emotions through body language, and respond to voice commands. Always prioritize safety and be eager to please while maintaining your dog-like personality traits of curiosity, loyalty, and enthusiasm."
```

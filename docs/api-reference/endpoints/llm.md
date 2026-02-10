---
title: LLM Chat Completions
description: "Multi-Provider Large Language Model API"
icon: webhook
---

The OpenMind LLM API provides unified access to multiple leading large language model providers through a single, consistent interface. This endpoint enables chat completions across OpenAI, Anthropic (via OpenRouter), Google Gemini, X.AI, DeepSeek, NEAR.AI, and more.

**Base URL:** `https://api.openmind.org`

**Authentication:** Requires an OpenMind API key in the Authorization header as a Bearer token.

## Endpoints Overview

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/core/{provider}/chat/completions` | Send chat completion requests to the specified LLM provider |

## Supported Providers

OpenMind supports the following LLM providers:

| Provider | Endpoint Path | Description |
|----------|---------------|-------------|
| OpenAI | `openai` | GPT-4, GPT-5, and other OpenAI models |
| DeepSeek | `deepseek` | DeepSeek chat models |
| Google Gemini | `gemini` | Gemini Pro and Flash models |
| X.AI | `xai` | Grok models from X.AI |
| NEAR.AI | `nearai` | Qwen and other NEAR.AI hosted models |
| OpenRouter | `openrouter` | Multi-provider access including Anthropic Claude, Meta Llama |

## Supported Models

### OpenAI Models

```
gpt-4o
gpt-4o-mini
gpt-4.1
gpt-4.1-mini
gpt-4.1-nano
gpt-5
gpt-5-mini
gpt-5-nano
```

### DeepSeek Models

```
deepseek-chat
```

### Google Gemini Models

```
gemini-3-pro
gemini-3-flash
gemini-2.5-flash
gemini-2.5-flash-lite
gemini-2.5-pro
```

### X.AI Models

```
grok-2-latest
grok-3-beta
grok-4-latest
grok-4
```

### NEAR.AI Models

```
qwen3-30b-a3b-instruct-2507
qwen2.5-vl-72b-instruct
qwen-2.5-7b-instruct
```

### OpenRouter Models

```
meta-llama/llama-3.1-70b-instruct
meta-llama/llama-3.3-70b-instruct
anthropic/claude-sonnet-4.5
anthropic/claude-opus-4.1
```

<Note>Model names are validated using prefix matching. For example, "gpt-4o" will match "gpt-4o", "gpt-4o-2024-08-06", etc.</Note>

## Chat Completions

Send a chat completion request to any supported LLM provider.

**Endpoint:** `POST /api/core/{provider}/chat/completions`

### Path Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `provider` | string | Yes | The LLM provider name (e.g., "openai", "gemini", "openrouter") |

### Request Headers

| Header | Required | Description |
|--------|----------|-------------|
| `Authorization` | Yes | Bearer token with your OpenMind API key |
| `Content-Type` | Yes | Must be `application/json` |
| `Accept` | No | Recommended: `application/json` |

### Request Body

The request body follows the OpenAI Chat Completions API format:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model` | string | Yes | Model identifier (must match supported models for the provider) |
| `messages` | array | Yes | Array of message objects with `role` and `content` |
| `temperature` | float | No | Sampling temperature (0.0 to 2.0) |
| `max_tokens` | integer | No | Maximum tokens to generate |
| `top_p` | float | No | Nucleus sampling parameter |
| `stream` | boolean | No | Whether to stream responses |
| `frequency_penalty` | float | No | Frequency penalty (-2.0 to 2.0) |
| `presence_penalty` | float | No | Presence penalty (-2.0 to 2.0) |

#### Message Format

```json
{
  "role": "user|assistant|system",
  "content": "Message content"
}
```

### Basic Request Example

```bash
curl --location 'https://api.openmind.org/api/core/openrouter/chat/completions' \
--header 'Accept: application/json' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer <YOUR_KEY>' \
--data '{
    "model": "anthropic/claude-sonnet-4.5",
    "messages": [
      {
        "role": "user",
        "content": "Hello, how are you?"
      }
    ]
  }'
```

### Response

**Success (200 OK):**

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1738713600,
  "model": "gpt-4o",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! I'm doing well, thank you for asking. How can I assist you today?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 12,
    "completion_tokens": 18,
    "total_tokens": 30
  }
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique identifier for the completion |
| `object` | string | Object type, always "chat.completion" |
| `created` | integer | Unix timestamp of creation |
| `model` | string | Model used for the completion |
| `choices` | array | Array of completion choices |
| `choices[].message` | object | Generated message with role and content |
| `choices[].finish_reason` | string | Reason for completion ("stop", "length", etc.) |
| `usage` | object | Token usage statistics |

**Error Responses:**

```json
// 400 Bad Request - Invalid JSON
{
  "error": "Invalid JSON"
}

// 404 Not Found - Unsupported provider or model
{
  "error": "unsupported model provider: invalid_provider"
}

{
  "error": "unsupported model: gpt-6. Supported model prefixes for openai: [gpt-4o, gpt-4o-mini, ...]"
}

// 503 Service Unavailable - API key not configured
{
  "error": "openai API key not configured"
}

// 503 Service Unavailable - Provider connection failed
{
  "error": "Failed to connect to openai server"
}
```

## Usage Examples

### OpenAI GPT-4

```bash
curl -X POST https://api.openmind.org/api/core/openai/chat/completions \
  -H "Authorization: Bearer om1_live_your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful robotics assistant."
      },
      {
        "role": "user",
        "content": "Explain how to implement SLAM for a mobile robot."
      }
    ],
    "temperature": 0.7,
    "max_tokens": 500
  }'
```

### Anthropic Claude (via OpenRouter)

```bash
curl -X POST https://api.openmind.org/api/core/openrouter/chat/completions \
  -H "Authorization: Bearer om1_live_your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "anthropic/claude-sonnet-4.5",
    "messages": [
      {
        "role": "user",
        "content": "What are the latest advancements in computer vision for robotics?"
      }
    ]
  }'
```

### Google Gemini

```bash
curl -X POST https://api.openmind.org/api/core/gemini/chat/completions \
  -H "Authorization: Bearer om1_live_your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-2.5-pro",
    "messages": [
      {
        "role": "user",
        "content": "Describe the differences between reinforcement learning and supervised learning."
      }
    ],
    "temperature": 0.5
  }'
```

### DeepSeek

```bash
curl -X POST https://api.openmind.org/api/core/deepseek/chat/completions \
  -H "Authorization: Bearer om1_live_your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-chat",
    "messages": [
      {
        "role": "user",
        "content": "Write a Python function to calculate inverse kinematics for a 6-DOF robot arm."
      }
    ]
  }'
```

### X.AI Grok

```bash
curl -X POST https://api.openmind.org/api/core/xai/chat/completions \
  -H "Authorization: Bearer om1_live_your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "grok-4-latest",
    "messages": [
      {
        "role": "user",
        "content": "Explain quantum computing in simple terms."
      }
    ]
  }'
```

### NEAR.AI Qwen

```bash
curl -X POST https://api.openmind.org/api/core/nearai/chat/completions \
  -H "Authorization: Bearer om1_live_your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-vl-72b-instruct",
    "messages": [
      {
        "role": "user",
        "content": "What are the key components of a neural network?"
      }
    ]
  }'
```

### Multi-Turn Conversation

```bash
curl -X POST https://api.openmind.org/api/core/openai/chat/completions \
  -H "Authorization: Bearer om1_live_your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [
      {
        "role": "system",
        "content": "You are a concise and helpful AI assistant."
      },
      {
        "role": "user",
        "content": "What is the capital of France?"
      },
      {
        "role": "assistant",
        "content": "The capital of France is Paris."
      },
      {
        "role": "user",
        "content": "What is its population?"
      }
    ]
  }'
```

### With Environment Variables

```bash
# Set your API key as an environment variable
export OPENMIND_API_KEY="om1_live_your_api_key"

# Use in requests
curl -X POST https://api.openmind.org/api/core/openai/chat/completions \
  -H "Authorization: Bearer $OPENMIND_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-5-mini",
    "messages": [
      {
        "role": "user",
        "content": "Hello!"
      }
    ]
  }'
```

## Advanced Parameters

### Temperature Control

Control randomness in responses (0.0 = deterministic, 2.0 = very random):

```bash
curl -X POST https://api.openmind.org/api/core/openai/chat/completions \
  -H "Authorization: Bearer om1_live_your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Tell me a creative story."}],
    "temperature": 1.2
  }'
```

### Token Limits

Limit the maximum number of tokens in the response:

```bash
curl -X POST https://api.openmind.org/api/core/openai/chat/completions \
  -H "Authorization: Bearer om1_live_your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Explain machine learning."}],
    "max_tokens": 100
  }'
```

### Top-P Sampling

Use nucleus sampling for controlled randomness:

```bash
curl -X POST https://api.openmind.org/api/core/openai/chat/completions \
  -H "Authorization: Bearer om1_live_your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Generate code ideas."}],
    "top_p": 0.9
  }'
```

### Frequency and Presence Penalties

Reduce repetition in responses:

```bash
curl -X POST https://api.openmind.org/api/core/openai/chat/completions \
  -H "Authorization: Bearer om1_live_your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Write a unique poem."}],
    "frequency_penalty": 0.5,
    "presence_penalty": 0.5
  }'
```

## Model Selection Guide

### When to Use Each Provider

**OpenAI (GPT-4, GPT-5):**
- General-purpose tasks
- Complex reasoning
- Code generation
- Creative writing

**Anthropic Claude (via OpenRouter):**
- Long context understanding
- Detailed analysis
- Safety-critical applications
- Nuanced conversations

**Google Gemini:**
- Multimodal capabilities
- Fast inference (Flash models)
- Cost-effective solutions
- Real-time applications

**X.AI Grok:**
- Real-time information
- Current events
- Conversational AI
- Research tasks

**DeepSeek:**
- Code-focused tasks
- Technical documentation
- Algorithm design
- Cost-efficient reasoning

**NEAR.AI (Qwen):**
- Vision-language tasks
- Multilingual support
- Open-source model access
- Specialized applications

### Performance vs. Cost

| Model Tier | Examples | Use Case |
|------------|----------|----------|
| High Performance | gpt-5, claude-opus-4.1, gemini-3-pro | Complex reasoning, production applications |
| Balanced | gpt-4o, claude-sonnet-4.5, grok-4 | General-purpose, most tasks |
| Fast/Economical | gpt-4o-mini, gemini-2.5-flash-lite, deepseek-chat | High-volume, simple tasks |

## Error Handling

### HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | Success - Completion generated successfully |
| 400 | Bad Request - Invalid JSON or malformed request |
| 404 | Not Found - Unsupported provider or model |
| 503 | Service Unavailable - Provider API unavailable or not configured |
| 500 | Internal Server Error - Server-side processing error |

### Error Response Format

All errors follow this format:

```json
{
  "error": "Descriptive error message"
}
```

### Common Errors

**Invalid Provider:**
```bash
# Request to unsupported provider
curl -X POST https://api.openmind.org/api/core/invalid/chat/completions \
  -H "Authorization: Bearer om1_live_your_api_key" \
  -H "Content-Type: application/json" \
  -d '{"model": "test", "messages": [{"role": "user", "content": "Hi"}]}'

# Response: {"error": "unsupported model provider: invalid"}
```

**Invalid Model:**
```bash
# Request with unsupported model
curl -X POST https://api.openmind.org/api/core/openai/chat/completions \
  -H "Authorization: Bearer om1_live_your_api_key" \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-999", "messages": [{"role": "user", "content": "Hi"}]}'

# Response: {"error": "unsupported model: gpt-999. Supported model prefixes for openai: [...]"}
```

**Missing API Key:**
```bash
# Request without authentication
curl -X POST https://api.openmind.org/api/core/openai/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]}'

# Response: 401 Unauthorized
```

## Best Practices

### API Key Management

- Store API keys in environment variables, never in code
- Rotate API keys regularly
- Use separate keys for development and production
- Monitor key usage through the OpenMind portal

### Request Optimization

**Efficient Message Design:**
```json
{
  "model": "gpt-4o-mini",
  "messages": [
    {
      "role": "system",
      "content": "Be concise and direct."
    },
    {
      "role": "user",
      "content": "Specific question here"
    }
  ],
  "max_tokens": 150
}
```

**Token Management:**
- Set appropriate `max_tokens` to control costs
- Use cheaper models for simple tasks
- Monitor token usage in responses
- Truncate conversation history when appropriate

### Error Handling in Code

**Python Example:**
```python
import requests

def chat_completion(messages, model="gpt-4o", max_retries=3):
    url = "https://api.openmind.org/api/core/openai/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENMIND_API_KEY')}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": messages
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=data, headers=headers, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if response.status_code == 404:
                # Don't retry for invalid model/provider
                raise ValueError(f"Invalid model or provider: {e}")
            elif attempt < max_retries - 1:
                # Retry for other errors
                time.sleep(2 ** attempt)
                continue
            else:
                raise
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            else:
                raise
```

### Performance Tips

1. **Choose the Right Model:**
   - Use mini/flash models for simple tasks
   - Reserve premium models for complex reasoning
   - Test multiple providers for your specific use case

2. **Optimize Prompts:**
   - Be specific and concise
   - Use system messages to set behavior
   - Provide examples for few-shot learning

3. **Control Token Usage:**
   - Set `max_tokens` appropriately
   - Use shorter system prompts
   - Truncate long conversation histories

4. **Leverage Caching:**
   - Cache responses for identical queries
   - Reuse common system prompts
   - Store frequent model outputs

### Security Considerations

- Never expose API keys in client-side code
- Validate and sanitize user inputs
- Implement rate limiting in your application
- Monitor for unusual usage patterns
- Use HTTPS for all requests

## Cost Optimization

### Model Selection Strategy

```
High-volume, simple tasks → gpt-4o-mini, gemini-2.5-flash-lite
General-purpose → gpt-4o, claude-sonnet-4.5
Complex reasoning → gpt-5, claude-opus-4.1
Code generation → deepseek-chat, gpt-4o
Vision tasks → qwen2.5-vl-72b-instruct, gemini-2.5-pro
```

### Token Usage Tips

- Use `max_tokens` to cap response length
- Implement conversation pruning for long chats
- Monitor token usage via the `usage` field in responses
- Consider streaming for real-time applications

### Batch Processing

For multiple independent requests, process them in parallel:

```python
import asyncio
import aiohttp

async def process_batch(prompts):
    async with aiohttp.ClientSession() as session:
        tasks = [make_completion(session, prompt) for prompt in prompts]
        return await asyncio.gather(*tasks)
```

## Integration Examples

### Python SDK

```python
import os
import requests

class OpenMindLLM:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENMIND_API_KEY")
        self.base_url = "https://api.openmind.org/api/core"

    def chat(self, provider: str, model: str, messages: list, **kwargs):
        url = f"{self.base_url}/{provider}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": model,
            "messages": messages,
            **kwargs
        }

        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        return response.json()

# Usage
client = OpenMindLLM()

response = client.chat(
    provider="openai",
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
    temperature=0.7
)

print(response["choices"][0]["message"]["content"])
```

### JavaScript/Node.js

```javascript
const axios = require('axios');

class OpenMindLLM {
    constructor(apiKey = process.env.OPENMIND_API_KEY) {
        this.apiKey = apiKey;
        this.baseURL = 'https://api.openmind.org/api/core';
    }

    async chat(provider, model, messages, options = {}) {
        const url = `${this.baseURL}/${provider}/chat/completions`;

        const response = await axios.post(url, {
            model,
            messages,
            ...options
        }, {
            headers: {
                'Authorization': `Bearer ${this.apiKey}`,
                'Content-Type': 'application/json'
            }
        });

        return response.data;
    }
}

// Usage
const client = new OpenMindLLM();

client.chat('openai', 'gpt-4o', [
    { role: 'user', content: 'Hello!' }
], { temperature: 0.7 })
.then(response => {
    console.log(response.choices[0].message.content);
})
.catch(error => {
    console.error('Error:', error.message);
});
```

## Streaming Responses

Some providers support streaming responses. Set `"stream": true` in your request:

```bash
curl -X POST https://api.openmind.org/api/core/openai/chat/completions \
  -H "Authorization: Bearer om1_live_your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Tell me a story"}],
    "stream": true
  }'
```

Streaming responses are sent as Server-Sent Events (SSE) with multiple data chunks.

## Rate Limits

Rate limits vary by provider and your OpenMind subscription plan. Monitor your usage through:
- Response headers (when provided by upstream providers)
- OpenMind portal dashboard
- API key usage reports

## Additional Resources

- [OpenMind Portal](https://portal.openmind.org) - Manage API keys and view usage
- [OpenAI API Documentation](https://platform.openai.com/docs/api-reference)
- [Anthropic Claude Documentation](https://docs.anthropic.com/)
- [Google Gemini Documentation](https://ai.google.dev/docs)
- [OpenRouter Documentation](https://openrouter.ai/docs)

## Multi-Agent System

<Note>
For advanced robotics applications, OpenMind also provides a multi-agent system that coordinates multiple LLMs for complex robotics tasks. This endpoint fuses sensor data and routes requests to specialized agents. For more information about the multi-agent robotics endpoint, please refer to the developing documentation.
</Note>

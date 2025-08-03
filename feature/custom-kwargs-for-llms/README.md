# OpenAI LLM Integration

## Custom Kwargs Support

The OpenAILLM class now supports custom keyword arguments (kwargs) for both initialization and request customization. This feature allows you to pass additional parameters to both the OpenAI client initialization and individual API calls.

### Usage Examples

#### Initializing with Custom Kwargs

from your_module import OpenAILLM, LLMConfig

Configure the LLM with custom client parameters
llm = OpenAILLM(
output_model=YourResponseModel,
config=LLMConfig(api_key="your-api-key"),
timeout=30, # Custom timeout for client
max_retries=3 # Custom retry configuration
)


#### Making Requests with Custom Parameters

Make a request with custom parameters
response = await llm.ask(
prompt="Your prompt here",
temperature=0.7, # Control randomness
top_p=0.9, # Nucleus sampling parameter
max_tokens=100 # Maximum response length
)


### Supported Parameters

#### Client Initialization Parameters
- `timeout`: Request timeout in seconds
- `max_retries`: Maximum number of retry attempts
- Any other parameters supported by the OpenAI AsyncClient

#### Request Parameters
- `temperature`: Controls randomness (0.0 to 1.0)
- `top_p`: Nucleus sampling parameter
- `max_tokens`: Maximum length of generated response
- `presence_penalty`: Penalty for new tokens based on presence in text
- `frequency_penalty`: Penalty for new tokens based on frequency in text
- Any other parameters supported by the OpenAI Chat Completions API

### Error Handling

The implementation includes proper error handling for both invalid parameters and API errors. All errors are logged for debugging purposes.

### Testing

Run the tests using pytest:

pytest test_openai_llm.py -v

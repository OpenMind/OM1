import pytest
from pydantic import BaseModel
from unittest.mock import AsyncMock, patch
from your_module import OpenAILLM, LLMConfig

class TestResponse(BaseModel):
    text: str
    confidence: float

@pytest.mark.asyncio
async def test_openai_llm_with_kwargs():
    # Test configuration
    config = LLMConfig(
        api_key="test-key",
        base_url="https://test-api.com"
    )

    # Custom kwargs for testing
    custom_kwargs = {
        "timeout": 30,
        "max_retries": 3
    }

    # Initialize LLM with custom kwargs
    llm = OpenAILLM(TestResponse, config, **custom_kwargs)

    # Mock the AsyncClient
    with patch('openai.AsyncClient') as mock_client:
        # Setup mock response
        mock_response = AsyncMock()
        mock_response.choices = [
            AsyncMock(
                message=AsyncMock(
                    content='{"text": "Test response", "confidence": 0.95}'
                )
            )
        ]
        
        mock_client.return_value.beta.chat.completions.parse = AsyncMock(
            return_value=mock_response
        )

        # Test ask method with additional kwargs
        response = await llm.ask(
            "Test prompt",
            temperature=0.7,
            top_p=0.9
        )

        # Verify the response
        assert isinstance(response, TestResponse)
        assert response.text == "Test response"
        assert response.confidence == 0.95

        # Verify that kwargs were passed correctly
        mock_client.assert_called_once_with(
            base_url="https://test-api.com",
            api_key="test-key",
            timeout=30,
            max_retries=3
        )

@pytest.mark.asyncio
async def test_openai_llm_invalid_kwargs():
    config = LLMConfig(api_key="test-key")
    
    # Test with invalid kwargs
    with pytest.raises(ValueError):
        OpenAILLM(TestResponse, config, invalid_param="invalid")

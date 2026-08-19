import pytest
from unittest.mock import patch, MagicMock
from llm.openrouter_client import OpenRouterClient


def test_openrouter_client_init():
    client = OpenRouterClient(api_key="test_key", model="test_model")
    assert client.api_key == "test_key"
    assert client.model == "test_model"


def test_openrouter_client_missing_key():
    client = OpenRouterClient(api_key="", model="test_model")
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY environment variable is not configured"):
        client.generate(query="What is solar power?", context="Solar power comes from the sun.")


def test_openrouter_client_empty_inputs():
    client = OpenRouterClient(api_key="test_key")
    with pytest.raises(ValueError, match="Query cannot be empty"):
        client.generate(query="", context="Context")
    with pytest.raises(ValueError, match="Context cannot be empty"):
        client.generate(query="Query", context="")


@patch("requests.post")
def test_openrouter_client_success(mock_post):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "Solar power is energy from the sun."
                }
            }
        ],
        "usage": {
            "prompt_tokens": 40,
            "completion_tokens": 10,
            "total_tokens": 50
        }
    }
    mock_post.return_value = mock_response

    client = OpenRouterClient(api_key="test_key", model="anthropic/claude-3.5-haiku")
    result = client.generate(
        query="What is solar power?",
        context="Solar power is energy from the sun."
    )

    assert result["response"] == "Solar power is energy from the sun."
    assert result["prompt_tokens"] == 40
    assert result["completion_tokens"] == 10
    assert result["total_tokens"] == 50
    assert result["model"] == "anthropic/claude-3.5-haiku"
    assert "latency_ms" in result

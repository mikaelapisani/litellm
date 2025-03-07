import json
import pytest
import requests
from unittest.mock import patch, MagicMock
from litellm.llms.snowflake.handler import SnowflakeCortexInferenceService
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.types.utils import ModelResponse

@pytest.fixture
def mock_snowflake_llm():
    """Fixture to create a test instance of SnowflakeCortexInferenceService."""
    return SnowflakeCortexInferenceService()


@pytest.fixture
def mock_oauth_token():
    """Mock OAuth token file content."""
    return "mock_oauth_token"


def test_get_auth_headers_oauth(mock_snowflake_llm, mock_oauth_token, tmp_path):
    """Test that OAuth headers are correctly generated."""
    token_path = tmp_path / "oauth_token"
    token_path.write_text(mock_oauth_token)  # Simulate token file

    mock_snowflake_llm.snowflake_token_path = str(token_path)
    headers = mock_snowflake_llm._get_auth_headers()

    assert headers["Authorization"] == f"Bearer {mock_oauth_token}"
    assert headers["X-Snowflake-Authorization-Token-Type"] == "OAUTH"


def test_get_auth_headers_jwt(mock_snowflake_llm):
    """Test that JWT headers are correctly generated."""
    mock_snowflake_llm.snowflake_authmethod = "jwt"

    with patch("snowflake_cortex_llm.JWTGenerator.get_token", return_value="mock_jwt_token"):
        headers = mock_snowflake_llm._get_auth_headers()

    assert headers["Authorization"] == "Bearer mock_jwt_token"
    assert headers["X-Snowflake-Authorization-Token-Type"] == "KEYPAIR_JWT"


@patch("requests.post")
def test_completion_success(mock_post, mock_snowflake_llm):
    """Test a successful completion request."""
    mock_response = MagicMock()
    mock_response.headers = {"Content-Type": "application/json"}
    mock_response.json.return_value = {
        "choices": [{"finish_reason": "stop", "index": 0, "message": {"content": "Hello", "role": "assistant"}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
    }
    mock_post.return_value = mock_response

    response = mock_snowflake_llm.completion(model="mistral-large2",
        base_url="https://api.snowflake.com/llm",
        snowflake_account="test_account",
        snowflake_service_user="test_user",
        snowflake_authmethod="oauth",
        snowflake_token_path="/tmp/fake_token",  # Mocked file path
        privatekey_password=None,
        api_key="fake_api_key",
        timeout=10,
        messages=[{"role": "user", "content": "Hi"}])

    assert isinstance(response, ModelResponse)
    assert response.choices[0]["message"]["content"] == "Hello"
    assert response.usage["total_tokens"] == 15


@patch("requests.post")
def test_completion_failure(mock_post, mock_snowflake_llm):
    """Test handling of an API failure."""
    mock_post.side_effect = requests.exceptions.RequestException("API failure")

    with pytest.raises(BaseLLMException) as exc:
        mock_snowflake_llm.completion(messages=[{"role": "user", "content": "Hi"}])

    assert "API failure" in str(exc.value)


def test_oauth_token_missing(mock_snowflake_llm):
    """Test error handling when OAuth token file is missing."""
    mock_snowflake_llm.snowflake_token_path = "/non/existent/token"

    with pytest.raises(BaseLLMException) as exc:
        mock_snowflake_llm._get_auth_headers()

    assert "OAuth token file not found" in str(exc.value)
from unittest.mock import patch, MagicMock
import json
import requests
import pytest

from litellm.types.utils import ModelResponse
from litellm.llms.snowflake.handler import SnowflakeCortexInferenceService
from litellm.llms.base_llm.chat.transformation import BaseLLMException


@pytest.fixture
def mock_snowflake_llm():
    """Fixture to create a test instance of SnowflakeCortexInferenceService."""
    return SnowflakeCortexInferenceService()


@pytest.fixture
def mock_oauth_token():
    """Mock OAuth token file content."""
    return "mock_oauth_token"


def test_get_auth_headers_oauth(
    mock_snowflake_llm,
    mock_oauth_token,
    tmp_path,
):
    """Test that OAuth headers are correctly generated."""
    token_path = tmp_path / "oauth_token"
    token_path.write_text(mock_oauth_token)  # Simulate token file

    mock_snowflake_llm.snowflake_token_path = str(token_path)
    headers = mock_snowflake_llm._get_auth_headers(
        "oauth",
        token_path,
        "mock_snowflake_account",
        "mock_snowflake_user",
        "mock_api_key",
        None,
    )

    assert headers["Authorization"] == f"Bearer {mock_oauth_token}"
    assert headers["X-Snowflake-Authorization-Token-Type"] == "OAUTH"


def test_oauth_token_missing(mock_snowflake_llm):
    """Test error handling when OAuth token file is missing."""
    snowflake_token_path = "/non/existent/token"

    with pytest.raises(BaseLLMException) as exc:
        mock_snowflake_llm._get_auth_headers(
            "oauth",
            snowflake_token_path,
            "mock_snowflake_account",
            "mock_snowflake_user",
            "mock_api_key",
            None,
        )

    assert "OAuth token file not found" in str(exc.value)


@patch("litellm.llms.snowflake.jwt_generator.JWTGenerator.__init__", return_value=None)
@patch(
    "litellm.llms.snowflake.jwt_generator.JWTGenerator.get_token",
    return_value="mock_jwt_token",
)
def test_get_auth_headers_jwt(_, _, mock_snowflake_llm):
    """Test that JWT headers are correctly generated without triggering real crypto operations."""

    headers = mock_snowflake_llm._get_auth_headers(
        "jwt", None, "mock_snowflake_account", "mock_snowflake_user", "mock_api_key", ""
    )

    assert headers["Authorization"] == "Bearer mock_jwt_token"
    assert headers["X-Snowflake-Authorization-Token-Type"] == "KEYPAIR_JWT"


@patch("requests.post")
def test_completion_success(mock_post, mock_snowflake_llm, tmp_path, mock_oauth_token):
    """Test a successful completion request."""
    mock_response = MagicMock()
    mock_response.headers = {"Content-Type": "text/event-stream"}
    mock_response.iter_lines.return_value = [
        b'data: {"choices": [{"delta": {"content": "Hello"}}], "usage": {"prompt_tokens":15, "completion_tokens": 40}}\n'
    ]

    mock_post.return_value = mock_response
    token_path = tmp_path / "oauth_token"
    token_path.write_text(mock_oauth_token)  # Simulate token file

    response = mock_snowflake_llm.completion(
        model="mistral-large2",
        base_url="https://api.snowflake.com/llm",
        logging_obj=None,
        snowflake_account="mock_snowflake_account",
        snowflake_service_user="mock_snowflake_user",
        snowflake_authmethod="oauth",
        snowflake_token_path=token_path,  # Mocked file path
        privatekey_password=None,
        api_key="fake_api_key",
        timeout=10,
        messages=[{"role": "user", "content": "Hi"}],
    )

    assert isinstance(response, ModelResponse)
    print("response:", response.choices)
    assert response.model == "mistral-large2"
    assert response.choices[0]["message"].content == "Hello"
    assert response.usage.total_tokens == 55


@patch("requests.post")
def test_completion_success_several_lines(
    mock_post, mock_snowflake_llm, tmp_path, mock_oauth_token
):
    """Test a successful completion request."""
    mock_response = MagicMock()
    mock_response.headers = {"Content-Type": "text/event-stream"}
    mock_response.iter_lines.return_value = [
        b'data: {"choices": [{"delta": {"content": "Hello."}}], "usage": {"prompt_tokens":15, "completion_tokens": 40}}\n',
        b'data: {"choices": [{"delta": {"content": " how can I help you?"}}], "usage": {"prompt_tokens":0, "completion_tokens": 1}}\n',
        b'data: {"choices": [{"delta": {"content": " I am here."}}], "usage": {"prompt_tokens":0, "completion_tokens": 2}}\n',
    ]

    mock_post.return_value = mock_response
    token_path = tmp_path / "oauth_token"
    token_path.write_text(mock_oauth_token)  # Simulate token file

    response = mock_snowflake_llm.completion(
        model="mistral-large2",
        base_url="https://api.snowflake.com/llm",
        logging_obj=None,
        snowflake_account="mock_snowflake_account",
        snowflake_service_user="mock_snowflake_user",
        snowflake_authmethod="oauth",
        snowflake_token_path=token_path,  # Mocked file path
        privatekey_password=None,
        api_key="fake_api_key",
        timeout=10,
        messages=[{"role": "user", "content": "Hi"}],
    )

    assert isinstance(response, ModelResponse)
    print("response:", response.choices)
    assert response.model == "mistral-large2"
    assert (
        response.choices[0]["message"].content
        == "Hello. how can I help you? I am here."
    )
    assert response.usage.total_tokens == 58


@patch("requests.post")
def test_completion_failure(mock_post, mock_snowflake_llm, tmp_path, mock_oauth_token):
    """Test handling of an API failure."""
    mock_post.side_effect = requests.exceptions.RequestException("API failure")
    token_path = tmp_path / "oauth_token"
    token_path.write_text(mock_oauth_token)  # Simulate token file
    with pytest.raises(BaseLLMException) as exc:
        mock_snowflake_llm.completion(
            model="mistral-large2",
            base_url="https://api.snowflake.com/llm",
            logging_obj=None,
            snowflake_account="mock_snowflake_account",
            snowflake_service_user="mock_snowflake_user",
            snowflake_authmethod="oauth",
            snowflake_token_path=token_path,  # Mocked file path
            privatekey_password=None,
            api_key="fake_api_key",
            timeout=10,
            messages=[{"role": "user", "content": "Hi"}],
        )
    assert "API failure" in str(exc.value)


@patch("requests.post")
def test_format_callback(mock_post, mock_snowflake_llm, tmp_path, mock_oauth_token):
    """Test a successful completion request."""
    token_path = tmp_path / "oauth_token"
    token_path.write_text(mock_oauth_token)  # Simulate token file

    # Mock the format callback function
    mock_format_callback = MagicMock(
        return_value=[{"role": "user", "content": "Modified message"}]
    )

    # Set it as the format callback
    mock_snowflake_llm.set_format_callback(mock_format_callback)

    # Mock API response
    mock_post.return_value.status_code = 200
    mock_post.return_value.headers = {"Content-Type": "application/json"}
    mock_post.return_value.json.return_value = {}

    # Input messages
    input_messages = [{"role": "user", "content": "Original message"}]

    mock_snowflake_llm.completion(
        model="mistral-large2",
        base_url="https://api.snowflake.com/llm",
        logging_obj=None,
        snowflake_account="mock_snowflake_account",
        snowflake_service_user="mock_snowflake_user",
        snowflake_authmethod="oauth",
        snowflake_token_path=token_path,  # Mocked file path
        privatekey_password=None,
        api_key="fake_api_key",
        timeout=10,
        messages=input_messages,
    )
    # Ensure the format callback was called
    mock_format_callback.assert_called_once_with(messages=input_messages)

    # Ensure that the modified messages were passed to the request payload
    request_payload = json.loads(mock_post.call_args[1]["data"])  # Extract payload data
    assert request_payload["messages"] == [
        {"role": "user", "content": "Modified message"}
    ]  # Validate transformation


@patch("requests.post")
def test_generate_id_callback(
    mock_post, mock_snowflake_llm, tmp_path, mock_oauth_token
):
    """Test a successful completion request."""
    mock_response = MagicMock()
    mock_response.headers = {"Content-Type": "text/event-stream"}
    mock_response.iter_lines.return_value = [
        b'data: {"choices": [{"delta": {"content": "Hello"}}], "usage": {"prompt_tokens":15, "completion_tokens": 40}}]\n'
    ]

    mock_post.return_value = mock_response
    token_path = tmp_path / "oauth_token"
    token_path.write_text(mock_oauth_token)  # Simulate token file

    # Mock the generate id function
    mock_generate_id_callback = MagicMock(return_value="id_value")

    # Set it as the generate id function
    mock_snowflake_llm.set_generate_id_callback(mock_generate_id_callback)

    response = mock_snowflake_llm.completion(
        model="mistral-large2",
        base_url="https://api.snowflake.com/llm",
        logging_obj=None,
        snowflake_account="mock_snowflake_account",
        snowflake_service_user="mock_snowflake_user",
        snowflake_authmethod="oauth",
        snowflake_token_path=token_path,  # Mocked file path
        privatekey_password=None,
        api_key="fake_api_key",
        timeout=10,
        messages=[{"role": "user", "content": "Hi"}],
    )

    # Ensure the format callback was called
    mock_generate_id_callback.assert_called_once()

    assert isinstance(response, ModelResponse)
    assert response.id == "chatcmpl-id_value"


@patch("requests.post")
def test_logging_obj(mock_post, mock_snowflake_llm, tmp_path, mock_oauth_token):
    """Test a successful completion request."""
    mock_response = MagicMock()
    mock_response.headers = {"Content-Type": "text/event-stream"}
    mock_response.iter_lines.return_value = [
        b'data: {"choices": [{"delta": {"content": "Hello"}}], "usage": {"prompt_tokens":15, "completion_tokens": 40}}]\n'
    ]

    mock_post.return_value = mock_response
    token_path = tmp_path / "oauth_token"
    token_path.write_text(mock_oauth_token)  # Simulate token file

    # Mock LiteLLMLoggingObj
    logging_obj = MagicMock()

    response = mock_snowflake_llm.completion(
        model="mistral-large2",
        base_url="https://api.snowflake.com/llm",
        logging_obj=logging_obj,
        snowflake_account="mock_snowflake_account",
        snowflake_service_user="mock_snowflake_user",
        snowflake_authmethod="oauth",
        snowflake_token_path=token_path,  # Mocked file path
        privatekey_password=None,
        api_key="fake_api_key",
        timeout=10,
        messages=[{"role": "user", "content": "Hi"}],
    )

    # Validate pre_call and post_call were invoked
    logging_obj.pre_call.assert_called_once()  # Ensure pre_call was called once
    logging_obj.post_call.assert_called_once()  # Ensure post_call was called once

    assert isinstance(response, ModelResponse)

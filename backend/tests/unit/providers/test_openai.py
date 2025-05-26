import pytest
import os
from unittest.mock import patch, AsyncMock, MagicMock

from ....providers.openai import OpenAIProvider
import openai # For error types and response model mocks

# Define the static model list for comparison
STATIC_MODEL_LIST = [
    {"id": "gpt-3.5-turbo", "name": "GPT-3.5 Turbo", "owned_by": "openai", "created_at": 0},
    {"id": "gpt-4", "name": "GPT-4", "owned_by": "openai", "created_at": 0},
    {"id": "gpt-4-turbo-preview", "name": "GPT-4 Turbo Preview", "owned_by": "openai", "created_at": 0}
]

@pytest.fixture
def mock_openai_client():
    mock_client = AsyncMock(spec=openai.AsyncOpenAI)
    mock_client.models = AsyncMock()
    mock_client.chat = AsyncMock()
    mock_client.chat.completions = AsyncMock()
    return mock_client

@pytest.fixture
def provider_details():
    return {"provider_id": "openai-test", "display_name": "OpenAI Test Provider"}

# --- Constructor Tests --- 

def test_openai_provider_init_with_api_key(provider_details):
    provider = OpenAIProvider(**provider_details, api_key="test_key_direct")
    assert provider.api_key == "test_key_direct"
    assert provider.client is not None
    assert isinstance(provider.client, openai.AsyncOpenAI)

@patch.dict(os.environ, {"OPENAI_API_KEY": "test_key_env"})
def test_openai_provider_init_with_env_api_key(provider_details):
    provider = OpenAIProvider(**provider_details)
    assert provider.api_key == "test_key_env"
    assert provider.client is not None
    assert isinstance(provider.client, openai.AsyncOpenAI)

@patch.dict(os.environ, {}, clear=True) # Ensure OPENAI_API_KEY is not set
def test_openai_provider_init_no_api_key(provider_details, capsys):
    provider = OpenAIProvider(**provider_details)
    assert provider.api_key is None
    assert provider.client is None
    captured = capsys.readouterr()
    assert "Warning: OpenAI API key not found" in captured.out

# --- Getter Tests --- 

def test_get_id(provider_details):
    provider = OpenAIProvider(**provider_details, api_key="dummy_key")
    assert provider.get_id() == provider_details["provider_id"]

def test_get_name(provider_details):
    provider = OpenAIProvider(**provider_details, api_key="dummy_key")
    assert provider.get_name() == provider_details["display_name"]

# --- list_models Tests --- 

@pytest.mark.asyncio
async def test_list_models_no_api_key(provider_details, capsys):
    # Ensure no env var is set for this specific test
    with patch.dict(os.environ, {}, clear=True):
        provider = OpenAIProvider(**provider_details)
    models = await provider.list_models()
    assert models == STATIC_MODEL_LIST
    captured = capsys.readouterr()
    # Initial warning from __init__ and then from list_models
    assert captured.out.count("Warning: OpenAI API key not found") >= 1
    assert "Warning: OpenAI client not initialized. Returning static list of models." in captured.out

@pytest.mark.asyncio
async def test_list_models_success(provider_details, mock_openai_client):
    # Mock the API response
    mock_model_data = [
        MagicMock(id="gpt-custom-1", owned_by="test_org", created=1678886400),
        MagicMock(id="gpt-custom-2", owned_by="openai", created=1678886401),
    ]
    mock_models_page = MagicMock()
    mock_models_page.data = mock_model_data
    mock_openai_client.models.list.return_value = mock_models_page

    with patch('openai.AsyncOpenAI', return_value=mock_openai_client):
        provider = OpenAIProvider(**provider_details, api_key="fake_key")
        models = await provider.list_models()

    expected_models = [
        {"id": "gpt-custom-1", "name": "gpt-custom-1", "owned_by": "test_org", "created_at": 1678886400},
        {"id": "gpt-custom-2", "name": "gpt-custom-2", "owned_by": "openai", "created_at": 1678886401},
    ]
    assert models == expected_models
    mock_openai_client.models.list.assert_awaited_once()

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "api_error",
    [
        openai.APIConnectionError(request=MagicMock()),
        openai.AuthenticationError(message="auth error", response=MagicMock(), body=None),
        openai.RateLimitError(message="rate limit", response=MagicMock(), body=None),
        openai.APIError(message="generic api error", request=MagicMock(), body=None)
    ]
)
async def test_list_models_api_errors(provider_details, mock_openai_client, api_error, capsys):
    mock_openai_client.models.list.side_effect = api_error

    with patch('openai.AsyncOpenAI', return_value=mock_openai_client):
        provider = OpenAIProvider(**provider_details, api_key="fake_key")
        models = await provider.list_models()

    assert models == STATIC_MODEL_LIST
    captured_out = capsys.readouterr().out
    error_name = type(api_error).__name__
    expected_msg_prefix = ""
    if error_name == "APIConnectionError":
        expected_msg_prefix = "OpenAI API Connection Error"
    elif error_name == "AuthenticationError":
        expected_msg_prefix = "OpenAI API Authentication Error"
    elif error_name == "RateLimitError":
        expected_msg_prefix = "OpenAI API Rate Limit Error"
    elif error_name == "APIError": # Base APIError
        expected_msg_prefix = "OpenAI API Error"
    else:
        pytest.fail(f"Unexpected error type {error_name} in test_list_models_api_errors parametrization")
    
    assert f"{expected_msg_prefix}: {str(api_error)}" in captured_out
    assert ". Returning static list of models." in captured_out
    mock_openai_client.models.list.assert_awaited_once()

# --- chat_stream Tests --- 

@pytest.mark.asyncio
async def test_chat_stream_no_api_key(provider_details):
    with patch.dict(os.environ, {}, clear=True):
        provider = OpenAIProvider(**provider_details)
    
    results = []
    async for event in provider.chat_stream(prompt="Hello", model_id="gpt-3.5-turbo"):
        results.append(event)
    
    assert len(results) == 1
    assert results[0] == {"error": "OpenAI API key not configured."}

@pytest.mark.asyncio
async def test_chat_stream_no_model_id(provider_details):
    provider = OpenAIProvider(**provider_details, api_key="fake_key")
    results = []
    async for event in provider.chat_stream(prompt="Hello"):
        results.append(event)
    
    assert len(results) == 1
    assert results[0] == {"error": "Model ID is required for OpenAI chat stream."}

@pytest.mark.asyncio
async def test_chat_stream_success(provider_details, mock_openai_client):
    # Mock the streaming response
    async def mock_stream_generator():
        yield MagicMock(choices=[MagicMock(delta=MagicMock(content="Hello"))])
        yield MagicMock(choices=[MagicMock(delta=MagicMock(content=" World"))])
        yield MagicMock(choices=[MagicMock(delta=MagicMock(content=None))]) # Empty content
        yield MagicMock(choices=[MagicMock(delta=MagicMock(content="!"))])

    mock_openai_client.chat.completions.create.return_value = mock_stream_generator()

    with patch('openai.AsyncOpenAI', return_value=mock_openai_client):
        provider = OpenAIProvider(**provider_details, api_key="fake_key")
        results = []
        async for event in provider.chat_stream(prompt="Test prompt", model_id="gpt-3.5-turbo", settings={"temperature": 0.5, "max_tokens": 100}):
            results.append(event)

    expected_tokens = [{"token": "Hello"}, {"token": " World"}, {"token": "!"}]
    assert results == expected_tokens
    mock_openai_client.chat.completions.create.assert_awaited_once_with(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Test prompt"}],
        stream=True,
        temperature=0.5,
        max_tokens=100
    )

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "api_error_type, error_message_detail",
    [
        (openai.APIConnectionError(request=MagicMock()), "APIConnectionError"),
        (openai.AuthenticationError(message="auth error", response=MagicMock(), body=None), "AuthenticationError"),
        (openai.RateLimitError(message="rate limit", response=MagicMock(), body=None), "RateLimitError"),
        (openai.BadRequestError(message="bad request", response=MagicMock(), body=None), "BadRequestError"),
        (openai.APIError(message="generic api error", request=MagicMock(), body=None), "APIError")
    ]
)
async def test_chat_stream_api_errors(provider_details, mock_openai_client, api_error_type, error_message_detail, capsys):
    mock_openai_client.chat.completions.create.side_effect = api_error_type

    with patch('openai.AsyncOpenAI', return_value=mock_openai_client):
        provider = OpenAIProvider(**provider_details, api_key="fake_key")
        results = []
        async for event in provider.chat_stream(prompt="Test prompt", model_id="gpt-3.5-turbo"):
            results.append(event)

    assert len(results) == 1
    assert "error" in results[0]
    # This asserts the content of the yielded error dictionary
    assert f"OpenAI API Error: {error_message_detail} - {str(api_error_type)}" in results[0]["error"]
    
    captured_out = capsys.readouterr().out
    # This asserts the content of the print statement
    expected_print_prefix = ""
    if error_message_detail == "APIConnectionError":
        expected_print_prefix = "OpenAI API Connection Error during chat stream"
    elif error_message_detail == "AuthenticationError":
        expected_print_prefix = "OpenAI API Authentication Error during chat stream"
    elif error_message_detail == "RateLimitError":
        expected_print_prefix = "OpenAI API Rate Limit Error during chat stream"
    elif error_message_detail == "BadRequestError":
        expected_print_prefix = "OpenAI API Bad Request Error during chat stream"
    elif error_message_detail == "APIError": # Base APIError
        expected_print_prefix = "OpenAI API Error during chat stream"
    else:
        pytest.fail(f"Unexpected error type string {error_message_detail} in test_chat_stream_api_errors parametrization")

    assert f"{expected_print_prefix}: {str(api_error_type)}" in captured_out
    mock_openai_client.chat.completions.create.assert_awaited_once()

@pytest.mark.asyncio
async def test_chat_stream_unexpected_error(provider_details, mock_openai_client, capsys):
    mock_openai_client.chat.completions.create.side_effect = Exception("Unexpected failure")

    with patch('openai.AsyncOpenAI', return_value=mock_openai_client):
        provider = OpenAIProvider(**provider_details, api_key="fake_key")
        results = []
        async for event in provider.chat_stream(prompt="Test prompt", model_id="gpt-3.5-turbo"):
            results.append(event)

    assert len(results) == 1
    assert "error" in results[0]
    assert "Unexpected error: Exception - Unexpected failure" in results[0]["error"]
    captured = capsys.readouterr()
    assert "Unexpected error during OpenAI chat stream: Unexpected failure" in captured.out

import pytest
from unittest.mock import patch, MagicMock
from aimodels.providers.ollama_interface import OllamaInterface
from httpx import ConnectError


@pytest.fixture(autouse=True)
def set_api_url_var(monkeypatch):
    """Fixture to set environment variables for tests."""
    monkeypatch.setenv("OLLAMA_API_URL", "http://localhost:11434")


def test_completion():
    """Test that completions request successfully."""

    user_greeting = "Howdy!"
    message_history = [{"role": "user", "content": user_greeting}]
    selected_model = "best-model-ever"
    chosen_temperature = 0.77
    response_text_content = "mocked-text-response-from-ollama-model"

    interface = OllamaInterface()
    mock_response = {"message": {"content": response_text_content}}

    with patch.object(
        interface.ollama_client, "chat", return_value=mock_response
    ) as mock_chat:
        response = interface.chat_completion_create(
            messages=message_history,
            model=selected_model,
            temperature=chosen_temperature,
        )

        mock_chat.assert_called_with(
            model=selected_model,
            messages=message_history,
            options={"temperature": chosen_temperature},
        )

        assert response.choices[0].message.content == response_text_content


def test_connection_error():
    """Test that any issues with connections raise a RuntimeError."""

    interface = OllamaInterface()
    mock_connect_error = ConnectError(
        "arbitrary-connect-error-message", request=MagicMock()
    )

    with patch.object(interface.ollama_client, "chat", side_effect=mock_connect_error):
        with pytest.raises(RuntimeError) as excinfo:
            interface.chat_completion_create(messages=[], model="any-model-we-want")

        assert str(excinfo.value) == interface._OLLAMA_STATUS_ERROR_MESSAGE

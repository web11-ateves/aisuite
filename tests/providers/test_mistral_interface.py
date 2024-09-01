import pytest
from unittest.mock import patch, MagicMock

from mistralai.models.chat_completion import ChatMessage

from aisuite.providers.mistral_interface import MistralInterface


@pytest.fixture(autouse=True)
def set_api_key_env_var(monkeypatch):
    """Fixture to set environment variables for tests."""
    monkeypatch.setenv("MISTRAL_API_KEY", "test-api-key")


def test_mistral_interface():
    """High-level test that the interface is initialized and chat completions are requested successfully."""

    user_greeting = "Hello!"
    message_history = [{"role": "user", "content": user_greeting}]
    selected_model = "our-favorite-model"
    chosen_temperature = 0.75
    response_text_content = "mocked-text-response-from-model"

    interface = MistralInterface()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message = MagicMock()
    mock_response.choices[0].message.content = response_text_content

    with patch.object(
        interface.mistral_client, "chat", return_value=mock_response
    ) as mock_create:
        response = interface.chat_completion_create(
            messages=message_history,
            model=selected_model,
            temperature=chosen_temperature,
        )

        transformed_message_history = [
            ChatMessage(role=message["role"], content=message["content"])
            for message in message_history
        ]

        mock_create.assert_called_with(
            messages=transformed_message_history,
            model=selected_model,
            temperature=chosen_temperature,
        )

        assert response.choices[0].message.content == response_text_content

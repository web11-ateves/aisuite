import pytest
from unittest.mock import patch, MagicMock
from aisuite.providers.anthropic_interface import AnthropicInterface


@pytest.fixture(autouse=True)
def set_api_key_env_var(monkeypatch):
    """Fixture to set environment variables for tests."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")


def test_anthropic_interface():
    """High-level test that the interface is initialized and chat completions are requested successfully."""

    user_greeting = "Hello!"
    message_history = [{"role": "user", "content": user_greeting}]
    selected_model = "our-favorite-model"
    chosen_temperature = 0.75
    response_text_content = "mocked-text-response-from-model"

    interface = AnthropicInterface()
    mock_response = MagicMock()
    mock_response.content = [MagicMock()]
    mock_response.content[0].text = response_text_content

    with patch.object(
        interface.anthropic_client.messages, "create", return_value=mock_response
    ) as mock_create:
        response = interface.chat_completion_create(
            messages=message_history,
            model=selected_model,
            temperature=chosen_temperature,
        )

        transformed_message_history = [
            {"role": "user", "content": [{"type": "text", "text": user_greeting}]},
        ]

        mock_create.assert_called_with(
            messages=transformed_message_history,
            model=selected_model,
            temperature=chosen_temperature,
            max_tokens=4096,
        )

        assert response.choices[0].message.content == response_text_content

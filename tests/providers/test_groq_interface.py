import pytest
from unittest.mock import patch, MagicMock
from aimodels.providers.groq_interface import GroqInterface


@pytest.fixture(autouse=True)
def set_api_key_env_var(monkeypatch):
    """Fixture to set environment variables for tests."""
    monkeypatch.setenv("GROQ_API_KEY", "test-api-key")


def test_groq_interface():
    """Test chat completions."""

    user_greeting = "Howdy!"
    message_history = [{"role": "user", "content": user_greeting}]
    selected_model = "best-groq-model"
    chosen_temperature = 0.75
    response_text_content = "mocked-text-response-from-model"

    interface = GroqInterface()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message = MagicMock()
    mock_response.choices[0].message.content = response_text_content

    with patch.object(
        interface.groq_client.chat.completions, "create", return_value=mock_response
    ) as mock_create:
        response = interface.chat_completion_create(
            messages=message_history,
            model=selected_model,
            temperature=chosen_temperature,
        )

        mock_create.assert_called_with(
            model=selected_model,
            messages=message_history,
            temperature=chosen_temperature,
        )

        assert response.choices[0].message.content == response_text_content

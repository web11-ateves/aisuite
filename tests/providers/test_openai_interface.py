import pytest
from unittest.mock import patch, MagicMock
from aimodels.providers.openai_interface import OpenAIInterface


@pytest.fixture(autouse=True)
def set_api_key_env_var(monkeypatch):
    """Set environment variables for tests."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")


def test_openai_interface():
    """Test chat completions."""

    user_greeting = "Hello Hello!"
    message_history = [{"role": "user", "content": user_greeting}]
    selected_model = "our-favorite-model"
    chosen_temperature = 0.9
    response_text_content = "mocked-text-response-from-model"

    interface = OpenAIInterface()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message = MagicMock()
    mock_response.choices[0].message.content = response_text_content

    with patch.object(
        interface.openai_client.chat.completions, "create", return_value=mock_response
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

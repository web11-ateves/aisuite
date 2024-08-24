import pytest
from unittest.mock import patch, MagicMock
from aimodels.providers.vertex_interface import VertexInterface
from vertexai.generative_models import Content, Part


@pytest.fixture(autouse=True)
def set_api_key_env_var(monkeypatch):
    """Fixture to set environment variables for tests."""
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "path-to-service-account-json")
    monkeypatch.setenv("VERTEX_PROJECT_ID", "vertex-project-id")
    monkeypatch.setenv("VERTEX_REGION", "us-central1")


def test_vertex_interface():
    """High-level test that the interface is initialized and chat completions are requested successfully."""

    user_greeting = "Hello!"
    message_history = [{"role": "user", "content": user_greeting}]
    selected_model = "our-favorite-model"

    interface = VertexInterface()

    with patch("vertexai.generative_models.GenerativeModel") as mock_model:
        mock_chat = MagicMock()
        mock_model.return_value.start_chat.return_value = mock_chat
        mock_chat.send_message.return_value = "Mocked response"

        response = interface.chat_completion_create(
            messages=message_history, model=selected_model
        )

        mock_model.assert_called_once_with(selected_model)
        mock_model.return_value.start_chat.assert_called_once()
        mock_chat.send_message.assert_called_once_with(user_greeting)
        assert response == "Mocked response"


def test_convert_openai_to_vertex_ai():
    interface = VertexInterface()
    message_history = [{"role": "user", "content": "Hello!"}]
    result = interface.convert_openai_to_vertex_ai(message_history)
    assert isinstance(result[0], Content)
    assert result[0].role == "user"
    assert len(result[0].parts) == 1
    assert isinstance(result[0].parts[0], Part)
    assert result[0].parts[0].text == "Hello!"


def test_transform_roles():
    interface = VertexInterface()

    messages = [
        {"role": "system", "content": "Vertex: system message = 1st user message."},
        {"role": "user", "content": "User message 1."},
        {"role": "assistant", "content": "Assistant message 1."},
    ]

    expected_output = [
        {"role": "user", "content": "Vertex: system message = 1st user message."},
        {"role": "user", "content": "User message 1."},
        {"role": "assistant", "content": "Assistant message 1."},
    ]

    result = interface.transform_roles(messages, from_role="system", to_role="user")

    assert result == expected_output

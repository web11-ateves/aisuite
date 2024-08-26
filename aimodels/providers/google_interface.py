"""The interface to Google's Vertex AI."""

import os
from aimodels.framework import ProviderInterface, ChatCompletionResponse


class GoogleInterface(ProviderInterface):
    """Implements the ProviderInterface for interacting with Google's Vertex AI."""

    def __init__(self):
        """Set up the Google AI client with a project ID."""
        import vertexai

        project_id = os.getenv("GOOGLE_PROJECT_ID")
        location = os.getenv("GOOGLE_REGION")
        app_creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

        if not project_id or not location or not app_creds_path:
            raise EnvironmentError(
                "Missing one or more required Google environment variables: "
                "GOOGLE_PROJECT_ID, GOOGLE_REGION, GOOGLE_APPLICATION_CREDENTIALS. "
                "Please refer to the setup guide: /guides/google.md."
            )

        vertexai.init(project=project_id, location=location)

    def chat_completion_create(self, messages=None, model=None, temperature=0):
        """Request chat completions from the Google AI API.

        Args:
        ----
            model (str): Identifies the specific provider/model to use.
            messages (list of dict): A list of message objects in chat history.

        Returns:
        -------
            The ChatCompletionResponse with the completion result.

        """
        from vertexai.generative_models import GenerativeModel, GenerationConfig

        without_system_messages = self.transform_roles(
            messages=messages, from_role="system", to_role="user"
        )

        with_model_roles = self.transform_roles(
            messages=without_system_messages, from_role="assistant", to_role="model"
        )

        final_message_history = self.convert_openai_to_vertex_ai(with_model_roles[:-1])
        last_message = with_model_roles[-1]["content"]

        model = GenerativeModel(
            model, generation_config=GenerationConfig(temperature=temperature)
        )

        chat = model.start_chat(history=final_message_history)
        response = chat.send_message(last_message)
        return self.convert_response_to_openai_format(response)

    def convert_openai_to_vertex_ai(self, messages):
        """Convert OpenAI messages to Google AI messages."""
        from vertexai.generative_models import Content, Part

        history = []
        for message in messages:
            role = message["role"]
            content = message["content"]
            parts = [Part.from_text(content)]
            history.append(Content(role=role, parts=parts))
        return history

    def transform_roles(self, messages, from_role, to_role):
        """Transform the roles in the messages to the desired role."""
        for message in messages:
            if message["role"] == from_role:
                message["role"] = to_role
        return messages

    def convert_response_to_openai_format(self, response):
        """Convert Google AI response to OpenAI's ChatCompletionResponse format."""
        openai_response = ChatCompletionResponse()
        openai_response.choices[0].message.content = (
            response.candidates[0].content.parts[0].text
        )
        return openai_response

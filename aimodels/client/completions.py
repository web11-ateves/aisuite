"""Completions is instantiated with a client and manages completion requests in chat sessions."""


class Completions:
    """Manage completion requests in chat sessions."""

    def __init__(self, topmost_instance):
        """Initialize a new Completions instance.

        Args:
        ----
            topmost_instance: The chat session's client instance (MultiFMClient).

        """
        self.topmost_instance = topmost_instance

    def create(self, model=None, temperature=0, messages=None):
        """Create a completion request using a specified provider/model combination.

        Args:
        ----
            model (str): The model identifier with format 'provider:model'.
            temperature (float): The sampling temperature.
            messages (list): A list of previous messages.

        Returns:
        -------
            The resulting completion.

        """
        interface, model_name = self.topmost_instance.get_provider_interface(model)

        return interface.chat_completion_create(
            messages,
            model=model_name,
            temperature=temperature,
        )

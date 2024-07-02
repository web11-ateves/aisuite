"""Chat is instantiated with a client and manages completions."""

from .completions import Completions


class Chat:
    """Manage chat sessions with multiple providers."""

    def __init__(self, topmost_instance):
        """Initialize a new Chat instance.

        Args:
        ----
            topmost_instance: The chat session's client instance (MultiFMClient).

        """
        self.topmost_instance = topmost_instance
        self.completions = Completions(topmost_instance)

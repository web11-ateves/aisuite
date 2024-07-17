"""MultiFMClient manages a Chat across multiple provider interfaces."""

from .chat import Chat
from ..providers import (
    AnthropicInterface,
    FireworksInterface,
    GroqInterface,
    MistralInterface,
    OllamaInterface,
    OpenAIInterface,
    ReplicateInterface,
    TogetherInterface,
    OctoInterface,
    AWSBedrockInterface,
)


class MultiFMClient:
    """Manages multiple provider interfaces."""

    _MODEL_FORMAT_ERROR_MESSAGE_TEMPLATE = (
        "Expected ':' in model identifier to specify provider:model. Got {model}."
    )
    _NO_FACTORY_ERROR_MESSAGE_TEMPLATE = (
        "Could not find factory to create interface for provider '{provider}'."
    )

    def __init__(self):
        """Initialize the MultiFMClient instance.

        Attributes
        ----------
            chat (Chat): The chat session.
            all_interfaces (dict): Stores interface instances by provider names.
            all_factories (dict): Maps provider names to their corresponding interfaces.

        """
        self.chat = Chat(self)
        self.all_interfaces = {}
        self.all_factories = {
            "anthropic": AnthropicInterface,
            "fireworks": FireworksInterface,
            "groq": GroqInterface,
            "mistral": MistralInterface,
            "ollama": OllamaInterface,
            "openai": OpenAIInterface,
            "replicate": ReplicateInterface,
            "together": TogetherInterface,
            "octo": OctoInterface,
            "aws": AWSBedrockInterface
        }

    def get_provider_interface(self, model):
        """Retrieve or create a provider interface based on a model identifier.

        Args:
        ----
            model (str): The model identifier in the format 'provider:model'.

        Raises:
        ------
            ValueError: If the model identifier does colon-separate provider and model.
            Exception: If no factory is found from the supplied model.

        Returns:
        -------
            The interface instance for the provider and the model name.

        """
        if ":" not in model:
            raise ValueError(
                self._MODEL_FORMAT_ERROR_MESSAGE_TEMPLATE.format(model=model)
            )

        model_parts = model.split(":", maxsplit=1)
        provider = model_parts[0]
        model_name = model_parts[1]

        if provider in self.all_interfaces:
            return self.all_interfaces[provider], model_name

        if provider not in self.all_factories:
            raise Exception(
                self._NO_FACTORY_ERROR_MESSAGE_TEMPLATE.format(provider=provider)
            )

        self.all_interfaces[provider] = self.all_factories[provider]()
        return self.all_interfaces[provider], model_name

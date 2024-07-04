import pytest
from aimodels.client.multi_fm_client import MultiFMClient, AnthropicInterface


def test_get_provider_interface_with_new_instance():
    """Test that get_provider_interface creates a new instance of the interface."""
    client = MultiFMClient()
    interface, model_name = client.get_provider_interface("anthropic:some-model:v1")
    assert isinstance(interface, AnthropicInterface)
    assert model_name == "some-model:v1"
    assert client.all_interfaces["anthropic"] == interface


def test_get_provider_interface_with_existing_instance():
    """Test that get_provider_interface returns an existing instance of the interface, if already created."""
    client = MultiFMClient()

    # New interface instance
    new_instance, _ = client.get_provider_interface("anthropic:some-model:v2")

    # Call twice, get same instance back
    same_instance, _ = client.get_provider_interface("anthropic:some-model:v2")

    assert new_instance is same_instance


def test_get_provider_interface_with_invalid_format():
    client = MultiFMClient()

    with pytest.raises(ValueError) as exc_info:
        client.get_provider_interface("invalid-model-no-colon")

    assert "Expected ':' in model identifier" in str(exc_info.value)


def test_get_provider_interface_with_unknown_interface():
    client = MultiFMClient()

    with pytest.raises(Exception) as exc_info:
        client.get_provider_interface("unknown-interface:some-model")

    assert (
        "Could not find factory to create interface for provider 'unknown-interface'"
        in str(exc_info.value)
    )

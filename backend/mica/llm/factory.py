"""
LLM Factory for MICA

Provides a unified interface to create LLM instances from different providers.
"""

from typing import Optional

from ..config import config
from .argo import ArgoLLM, get_available_argo_models
from .base import BaseLLM
from .gemini import GeminiLLM, get_available_gemini_models


def create_llm(
    provider: Optional[str] = None,
    model_id: Optional[str] = None,
    **kwargs,
) -> BaseLLM:
    """
    Create an LLM instance based on provider and model.

    Args:
        provider: LLM provider ('argo' or 'gemini'). Defaults to config.
        model_id: Model ID. Defaults to config.
        **kwargs: Additional provider-specific parameters

    Returns:
        BaseLLM instance

    Raises:
        ValueError: If provider is unknown or configuration is missing

    Example:
        # Use default configuration
        llm = create_llm()

        # Specify provider and model
        llm = create_llm(provider="argo", model_id="claudesonnet45")

        # Use Gemini with custom temperature
        llm = create_llm(provider="gemini", model_id="gemini-flash", temperature=0.5)
    """
    # Use defaults from config if not specified
    provider = provider or config.llm.llm_provider
    model_id = model_id or config.llm.default_model

    if provider == "argo":
        username = kwargs.pop("username", None) or config.argo.username
        if not username:
            raise ValueError(
                "Argo username required. Set ARGO_USERNAME environment variable."
            )
        return ArgoLLM(model_id=model_id, username=username, **kwargs)

    elif provider == "gemini":
        api_key = kwargs.pop("api_key", None) or config.gemini.api_key
        if not api_key:
            raise ValueError(
                "Google API key required. Set GOOGLE_API_KEY environment variable."
            )
        return GeminiLLM(model_id=model_id, api_key=api_key, **kwargs)

    else:
        raise ValueError(
            f"Unknown provider: {provider}. Available: 'argo', 'gemini'"
        )


def create_llm_from_config() -> BaseLLM:
    """
    Create an LLM instance using the current configuration.

    This is a convenience function that reads all settings from config.

    Returns:
        BaseLLM instance configured from environment

    Raises:
        ValueError: If configuration is incomplete
    """
    llm_config = config.get_active_llm_config()

    if llm_config["provider"] == "argo":
        return ArgoLLM(
            model_id=llm_config["model_id"],
            username=llm_config["username"],
        )
    elif llm_config["provider"] == "gemini":
        return GeminiLLM(
            model_id=llm_config["model_id"],
            api_key=llm_config["api_key"],
        )
    else:
        raise ValueError(f"Unknown provider: {llm_config['provider']}")


def get_available_models() -> dict[str, dict[str, str]]:
    """
    Get all available models organized by provider.

    Returns:
        Dictionary with provider names as keys and model dictionaries as values

    Example:
        {
            "argo": {
                "claudesonnet45": "Claude Sonnet 4.5",
                "gpt4o": "GPT-4o",
                ...
            },
            "gemini": {
                "gemini-flash": "Gemini 2.0 Flash",
                ...
            }
        }
    """
    return {
        "argo": get_available_argo_models(),
        "gemini": get_available_gemini_models(),
    }


def validate_model(provider: str, model_id: str) -> bool:
    """
    Check if a model is valid for a provider.

    Args:
        provider: Provider name
        model_id: Model ID to validate

    Returns:
        True if valid, False otherwise
    """
    available = get_available_models()
    if provider not in available:
        return False
    return model_id in available[provider]

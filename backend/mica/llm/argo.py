"""
Argo LLM Provider for MICA

Provides access to LLMs via the Argonne Argo API.
Adapted from useful_utils/argo.py for LangGraph compatibility.
"""

import asyncio
from typing import Optional

import httpx
import requests
import tiktoken
from langchain_core.messages import AIMessage, BaseMessage

from ..config import config
from .base import BaseLLM, LLMResponse

# Available models via Argo API
ARGO_MODELS = {
    "gpt4o": {"name": "GPT-4o", "env": "prod"},
    "claudeopus4": {"name": "Claude Opus 4", "env": "dev"},
    "claudeopus45": {"name": "Claude Opus 4.5", "env": "dev"},
    "claudesonnet4": {"name": "Claude Sonnet 4", "env": "dev"},
    "claudesonnet45": {"name": "Claude Sonnet 4.5", "env": "dev"},
    "gemini25flash": {"name": "Gemini 2.5 Flash", "env": "dev"},
    "gpt35": {"name": "GPT-3.5", "env": "prod"},
    "gpt35large": {"name": "GPT-3.5 Large", "env": "prod"},
    "gpt4": {"name": "GPT-4", "env": "prod"},
    "gpt41": {"name": "GPT-4.1", "env": "dev"},
    "gpt4large": {"name": "GPT-4 Large", "env": "prod"},
    "gpt4turbo": {"name": "GPT-4 Turbo", "env": "prod"},
    "gpto1preview": {"name": "GPT-o1 Preview", "env": "prod"},
    "gemini25pro": {"name": "Gemini 2.5 Pro", "env": "dev"},
    "gpt5": {"name": "GPT-5", "env": "dev"},
    "gpt5mini": {"name": "GPT-5 Mini", "env": "dev"},
}

# API endpoints
ARGO_URLS = {
    "prod": "https://apps.inside.anl.gov/argoapi/api/v1/resource/chat/",
    "dev": "https://apps-dev.inside.anl.gov/argoapi/api/v1/resource/chat/",
    "test": "https://apps-test.inside.anl.gov/argoapi/api/v1/resource/chat/",
}

# Default timeout in seconds (3 minutes for complex queries)
DEFAULT_TIMEOUT = 180


def _count_tokens(text: str) -> int:
    """Count tokens using tiktoken."""
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(str(text)))
    except Exception:
        # Fallback: rough estimate
        return len(str(text)) // 4


def _get_api_url(model_id: str) -> str:
    """Get the appropriate API URL for a model."""
    model_info = ARGO_MODELS.get(model_id)
    if model_info:
        env = model_info.get("env", "prod")
        return ARGO_URLS.get(env, ARGO_URLS["prod"])
    return ARGO_URLS["prod"]


class ArgoLLM(BaseLLM):
    """
    Argo LLM provider for MICA.

    Connects to the Argonne Argo API to access various LLM models.
    """

    def __init__(
        self,
        model_id: str = "claudesonnet45",
        username: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 20000,
        **kwargs,
    ):
        """
        Initialize Argo LLM.

        Args:
            model_id: The model to use (e.g., 'claudesonnet45', 'claudeopus45', 'gpt4o')
            username: Argo username (defaults to ARGO_USERNAME env var)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
        """
        super().__init__(model_id, **kwargs)

        self.username = username or config.argo.username
        if not self.username:
            raise ValueError(
                "Argo username required. Set ARGO_USERNAME environment variable "
                "or pass username parameter."
            )

        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_url = _get_api_url(model_id)

    @property
    def provider_name(self) -> str:
        return "argo"

    def _build_request_data(
        self,
        messages: list[dict],
        stop: Optional[list[str]] = None,
    ) -> dict:
        """Build the request payload for Argo API."""
        data = {
            "user": self.username,
            "model": self.model_id,
            "messages": messages,
            "stop": stop or [],
        }

        # Model-specific parameters
        if self.model_id == "claudesonnet45":
            # Claude Sonnet 4.5 only accepts temperature, not top_p
            data["temperature"] = self.temperature
            data["max_tokens"] = self.max_tokens
        elif self.model_id.startswith("claude"):
            data["temperature"] = self.temperature
            data["top_p"] = 1.0
            data["max_tokens"] = self.max_tokens
        elif self.model_id in ["gpt5", "gpt5mini"]:
            data["temperature"] = self.temperature
            data["top_p"] = 0.9
            data["max_completion_tokens"] = self.max_tokens
        elif self.model_id.startswith("gpto") or self.model_id.startswith("gpt4"):
            data["max_completion_tokens"] = self.max_tokens
        else:
            data["temperature"] = self.temperature
            data["top_p"] = 1.0
            data["max_tokens"] = self.max_tokens

        return data

    def _convert_messages(self, messages: list[BaseMessage]) -> list[dict]:
        """Convert LangChain messages to Argo API format."""
        argo_messages = []
        for msg in self._convert_messages_to_dicts(messages):
            role = msg["role"]
            content = msg["content"]

            # Handle o-series models that don't support system messages
            if role == "system" and self.model_id in ["gpto1preview", "gpto1mini", "gpto1"]:
                role = "user"
                content = f"System: {content}"

            argo_messages.append({"role": role, "content": content})
        return argo_messages

    def _parse_response(self, response_json: dict) -> str:
        """Parse the response from Argo API."""
        if isinstance(response_json, dict):
            text = response_json.get("response")
            if text is None:
                text = (
                    response_json.get("content")
                    or response_json.get("message")
                    or response_json.get("text")
                )
            if text is None:
                raise ValueError(
                    f"No response content in API response. Keys: {list(response_json.keys())}"
                )
            return text
        return str(response_json)

    def invoke(self, messages: list[BaseMessage], **kwargs) -> AIMessage:
        """
        Invoke the LLM synchronously.

        Args:
            messages: List of LangChain messages
            **kwargs: Additional parameters

        Returns:
            AIMessage with the response
        """
        argo_messages = self._convert_messages(messages)
        data = self._build_request_data(argo_messages, kwargs.get("stop"))

        try:
            response = requests.post(
                self.api_url,
                headers={"Content-Type": "application/json"},
                json=data,
                timeout=kwargs.get("timeout", DEFAULT_TIMEOUT),
            )

            if response.status_code != 200:
                error_msg = self._format_error(response)
                return AIMessage(content=f"Error: {error_msg}")

            result = response.json()
            content = self._parse_response(result)

            # Calculate token counts
            input_text = " ".join(m["content"] for m in argo_messages)
            input_tokens = _count_tokens(input_text)
            output_tokens = _count_tokens(content)

            return AIMessage(
                content=content,
                additional_kwargs={
                    "model": self.model_id,
                    "provider": self.provider_name,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                },
            )

        except requests.exceptions.Timeout:
            return AIMessage(
                content=f"Error: Request timeout after {DEFAULT_TIMEOUT}s. "
                f"Model: {self.model_id}"
            )
        except requests.exceptions.RequestException as e:
            return AIMessage(content=f"Error: Request failed - {str(e)}")
        except Exception as e:
            return AIMessage(content=f"Error: {type(e).__name__}: {str(e)}")

    async def ainvoke(self, messages: list[BaseMessage], **kwargs) -> AIMessage:
        """
        Invoke the LLM asynchronously.

        Args:
            messages: List of LangChain messages
            **kwargs: Additional parameters

        Returns:
            AIMessage with the response
        """
        argo_messages = self._convert_messages(messages)
        data = self._build_request_data(argo_messages, kwargs.get("stop"))

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.api_url,
                    headers={"Content-Type": "application/json"},
                    json=data,
                    timeout=kwargs.get("timeout", DEFAULT_TIMEOUT),
                )

                if response.status_code != 200:
                    error_msg = self._format_error_httpx(response)
                    return AIMessage(content=f"Error: {error_msg}")

                result = response.json()
                content = self._parse_response(result)

                input_text = " ".join(m["content"] for m in argo_messages)
                input_tokens = _count_tokens(input_text)
                output_tokens = _count_tokens(content)

                return AIMessage(
                    content=content,
                    additional_kwargs={
                        "model": self.model_id,
                        "provider": self.provider_name,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                    },
                )

        except httpx.TimeoutException:
            return AIMessage(
                content=f"Error: Request timeout after {DEFAULT_TIMEOUT}s. "
                f"Model: {self.model_id}"
            )
        except Exception as e:
            return AIMessage(content=f"Error: {type(e).__name__}: {str(e)}")

    def _format_error(self, response: requests.Response) -> str:
        """Format error message from requests response."""
        status = response.status_code
        if status == 401:
            return f"Authentication failed. Check ARGO_USERNAME: '{self.username}'"
        elif status == 403:
            return f"Access forbidden for user '{self.username}' on model '{self.model_id}'"
        elif status == 404:
            return f"Model '{self.model_id}' not found"
        elif status == 500:
            return "Argo API internal error. Try again later."
        else:
            return f"HTTP {status}: {response.text[:200]}"

    def _format_error_httpx(self, response: httpx.Response) -> str:
        """Format error message from httpx response."""
        status = response.status_code
        if status == 401:
            return f"Authentication failed. Check ARGO_USERNAME: '{self.username}'"
        elif status == 403:
            return f"Access forbidden for user '{self.username}' on model '{self.model_id}'"
        elif status == 404:
            return f"Model '{self.model_id}' not found"
        elif status == 500:
            return "Argo API internal error. Try again later."
        else:
            return f"HTTP {status}: {response.text[:200]}"


def get_available_argo_models() -> dict[str, str]:
    """Get dictionary of available Argo models with descriptions."""
    return {model_id: info["name"] for model_id, info in ARGO_MODELS.items()}

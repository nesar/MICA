"""
Google Gemini LLM Provider for MICA

Provides access to Google's Gemini models via the Generative AI API.
"""

from typing import Optional

import httpx
import requests
from langchain_core.messages import AIMessage, BaseMessage

from ..config import config
from .base import BaseLLM

# Available Gemini models
GEMINI_MODELS = {
    "gemini-flash": {
        "id": "gemini-2.0-flash",
        "name": "Gemini 2.0 Flash",
    },
    "gemini-thinking": {
        "id": "gemini-2.0-flash-thinking-exp-1219",
        "name": "Gemini 2.0 Flash Thinking",
    },
    "gemini-flash25": {
        "id": "gemini-2.5-flash",
        "name": "Gemini 2.5 Flash",
    },
}

# Gemini API base URL (OpenAI-compatible endpoint)
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/openai"

DEFAULT_TIMEOUT = 120


class GeminiLLM(BaseLLM):
    """
    Google Gemini LLM provider for MICA.

    Uses the OpenAI-compatible endpoint for Gemini models.
    """

    def __init__(
        self,
        model_id: str = "gemini-flash",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 8192,
        **kwargs,
    ):
        """
        Initialize Gemini LLM.

        Args:
            model_id: The model to use (e.g., 'gemini-flash', 'gemini-flash25')
            api_key: Google API key (defaults to GOOGLE_API_KEY env var)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
        """
        super().__init__(model_id, **kwargs)

        self.api_key = api_key or config.gemini.api_key
        if not self.api_key:
            raise ValueError(
                "Google API key required. Set GOOGLE_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.temperature = temperature
        self.max_tokens = max_tokens

        # Resolve model ID to actual Gemini model name
        model_info = GEMINI_MODELS.get(model_id)
        if model_info:
            self.gemini_model_id = model_info["id"]
        else:
            # Assume it's already the full model ID
            self.gemini_model_id = model_id

    @property
    def provider_name(self) -> str:
        return "gemini"

    def _get_api_url(self) -> str:
        """Get the API endpoint URL."""
        return f"{GEMINI_API_BASE}/chat/completions"

    def _build_request_data(
        self,
        messages: list[dict],
        stop: Optional[list[str]] = None,
    ) -> dict:
        """Build the request payload for Gemini API."""
        data = {
            "model": self.gemini_model_id,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        if stop:
            data["stop"] = stop

        return data

    def _get_headers(self) -> dict:
        """Get request headers."""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _parse_response(self, response_json: dict) -> str:
        """Parse the response from Gemini API (OpenAI format)."""
        try:
            choices = response_json.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                return message.get("content", "")
            return ""
        except Exception as e:
            raise ValueError(f"Failed to parse response: {e}")

    def _extract_usage(self, response_json: dict) -> tuple[int, int]:
        """Extract token usage from response."""
        usage = response_json.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        return input_tokens, output_tokens

    def invoke(self, messages: list[BaseMessage], **kwargs) -> AIMessage:
        """
        Invoke the LLM synchronously.

        Args:
            messages: List of LangChain messages
            **kwargs: Additional parameters

        Returns:
            AIMessage with the response
        """
        formatted_messages = self._convert_messages_to_dicts(messages)
        data = self._build_request_data(formatted_messages, kwargs.get("stop"))

        try:
            response = requests.post(
                self._get_api_url(),
                headers=self._get_headers(),
                json=data,
                timeout=kwargs.get("timeout", DEFAULT_TIMEOUT),
            )

            if response.status_code != 200:
                error_msg = self._format_error(response)
                return AIMessage(content=f"Error: {error_msg}")

            result = response.json()
            content = self._parse_response(result)
            input_tokens, output_tokens = self._extract_usage(result)

            return AIMessage(
                content=content,
                additional_kwargs={
                    "model": self.gemini_model_id,
                    "provider": self.provider_name,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                },
            )

        except requests.exceptions.Timeout:
            return AIMessage(
                content=f"Error: Request timeout after {DEFAULT_TIMEOUT}s. "
                f"Model: {self.gemini_model_id}"
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
        formatted_messages = self._convert_messages_to_dicts(messages)
        data = self._build_request_data(formatted_messages, kwargs.get("stop"))

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self._get_api_url(),
                    headers=self._get_headers(),
                    json=data,
                    timeout=kwargs.get("timeout", DEFAULT_TIMEOUT),
                )

                if response.status_code != 200:
                    error_msg = self._format_error_httpx(response)
                    return AIMessage(content=f"Error: {error_msg}")

                result = response.json()
                content = self._parse_response(result)
                input_tokens, output_tokens = self._extract_usage(result)

                return AIMessage(
                    content=content,
                    additional_kwargs={
                        "model": self.gemini_model_id,
                        "provider": self.provider_name,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                    },
                )

        except httpx.TimeoutException:
            return AIMessage(
                content=f"Error: Request timeout after {DEFAULT_TIMEOUT}s. "
                f"Model: {self.gemini_model_id}"
            )
        except Exception as e:
            return AIMessage(content=f"Error: {type(e).__name__}: {str(e)}")

    def _format_error(self, response: requests.Response) -> str:
        """Format error message from requests response."""
        status = response.status_code
        try:
            error_data = response.json()
            error_msg = error_data.get("error", {}).get("message", response.text[:200])
        except Exception:
            error_msg = response.text[:200]

        if status == 401:
            return f"Authentication failed. Check GOOGLE_API_KEY."
        elif status == 403:
            return f"Access forbidden. API key may not have access to {self.gemini_model_id}"
        elif status == 404:
            return f"Model '{self.gemini_model_id}' not found"
        elif status == 429:
            return "Rate limit exceeded. Try again later."
        else:
            return f"HTTP {status}: {error_msg}"

    def _format_error_httpx(self, response: httpx.Response) -> str:
        """Format error message from httpx response."""
        status = response.status_code
        try:
            error_data = response.json()
            error_msg = error_data.get("error", {}).get("message", response.text[:200])
        except Exception:
            error_msg = response.text[:200]

        if status == 401:
            return f"Authentication failed. Check GOOGLE_API_KEY."
        elif status == 403:
            return f"Access forbidden. API key may not have access to {self.gemini_model_id}"
        elif status == 404:
            return f"Model '{self.gemini_model_id}' not found"
        elif status == 429:
            return "Rate limit exceeded. Try again later."
        else:
            return f"HTTP {status}: {error_msg}"


def get_available_gemini_models() -> dict[str, str]:
    """Get dictionary of available Gemini models with descriptions."""
    return {model_id: info["name"] for model_id, info in GEMINI_MODELS.items()}

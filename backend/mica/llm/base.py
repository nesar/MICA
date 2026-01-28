"""
Base LLM Interface for MICA

Provides a unified interface for different LLM providers that works with LangGraph.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage


@dataclass
class LLMResponse:
    """Standardized response from LLM calls."""

    content: str
    model: str
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0
    metadata: dict = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        """Total tokens used in this response."""
        return self.input_tokens + self.output_tokens


class BaseLLM(ABC):
    """
    Abstract base class for LLM providers in MICA.

    Provides a unified interface for different LLM backends while supporting
    LangGraph's message-based interaction patterns.
    """

    def __init__(self, model_id: str, **kwargs):
        """
        Initialize the LLM provider.

        Args:
            model_id: The specific model to use (e.g., 'claudeopus4', 'gemini-flash')
            **kwargs: Provider-specific configuration
        """
        self.model_id = model_id
        self.kwargs = kwargs

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name (e.g., 'argo', 'gemini')."""
        pass

    @abstractmethod
    def invoke(self, messages: list[BaseMessage], **kwargs) -> AIMessage:
        """
        Invoke the LLM with a list of messages.

        Args:
            messages: List of LangChain messages
            **kwargs: Additional parameters for this call

        Returns:
            AIMessage with the response
        """
        pass

    @abstractmethod
    async def ainvoke(self, messages: list[BaseMessage], **kwargs) -> AIMessage:
        """
        Async version of invoke.

        Args:
            messages: List of LangChain messages
            **kwargs: Additional parameters for this call

        Returns:
            AIMessage with the response
        """
        pass

    def get_langchain_llm(self) -> BaseLanguageModel:
        """
        Get a LangChain-compatible LLM instance.

        This is used for LangGraph integration.
        """
        return LangChainLLMWrapper(self)

    def _convert_messages_to_dicts(self, messages: list[BaseMessage]) -> list[dict]:
        """Convert LangChain messages to dictionary format."""
        result = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                result.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                result.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                result.append({"role": "assistant", "content": msg.content})
            else:
                # Default to user message for unknown types
                result.append({"role": "user", "content": str(msg.content)})
        return result


class LangChainLLMWrapper(BaseLanguageModel):
    """
    Wrapper to make BaseLLM compatible with LangChain/LangGraph.

    This allows our custom LLM implementations to work seamlessly with
    LangGraph's agent and workflow patterns.
    """

    def __init__(self, llm: BaseLLM):
        """
        Initialize the wrapper.

        Args:
            llm: The MICA LLM instance to wrap
        """
        super().__init__()
        self._llm = llm

    @property
    def _llm_type(self) -> str:
        """Return the LLM type identifier."""
        return f"mica-{self._llm.provider_name}"

    def _generate(self, prompts: list[str], stop: Optional[list[str]] = None, **kwargs) -> Any:
        """Generate responses for prompts (not used in chat models)."""
        raise NotImplementedError("Use invoke() with messages instead")

    def invoke(self, input: Any, config: Optional[dict] = None, **kwargs) -> AIMessage:
        """
        Invoke the LLM.

        Args:
            input: Messages or prompt
            config: Optional configuration
            **kwargs: Additional parameters

        Returns:
            AIMessage with the response
        """
        # Handle different input types
        if isinstance(input, list):
            messages = input
        elif isinstance(input, str):
            messages = [HumanMessage(content=input)]
        else:
            messages = [HumanMessage(content=str(input))]

        return self._llm.invoke(messages, **kwargs)

    async def ainvoke(self, input: Any, config: Optional[dict] = None, **kwargs) -> AIMessage:
        """
        Async invoke the LLM.

        Args:
            input: Messages or prompt
            config: Optional configuration
            **kwargs: Additional parameters

        Returns:
            AIMessage with the response
        """
        if isinstance(input, list):
            messages = input
        elif isinstance(input, str):
            messages = [HumanMessage(content=input)]
        else:
            messages = [HumanMessage(content=str(input))]

        return await self._llm.ainvoke(messages, **kwargs)

    def _call(self, prompt: str, stop: Optional[list[str]] = None, **kwargs) -> str:
        """Simple call interface for compatibility."""
        messages = [HumanMessage(content=prompt)]
        response = self._llm.invoke(messages, **kwargs)
        return response.content

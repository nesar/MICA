"""
MICA Configuration Management

Centralized configuration using environment variables with Pydantic settings.
All configuration values are loaded from environment variables or .env file.
"""

import os
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    """LLM provider configuration."""

    model_config = SettingsConfigDict(env_prefix="MICA_", extra="ignore")

    llm_provider: Literal["argo", "gemini"] = Field(
        default="argo",
        description="LLM provider to use",
    )
    default_model: str = Field(
        default="claudeopus4",
        description="Default model ID",
    )


class ArgoSettings(BaseSettings):
    """Argo API configuration."""

    model_config = SettingsConfigDict(extra="ignore")

    username: Optional[str] = Field(
        default=None,
        alias="ARGO_USERNAME",
        description="Argo API username",
    )

    @property
    def is_configured(self) -> bool:
        """Check if Argo is properly configured."""
        return self.username is not None and len(self.username) > 0


class GeminiSettings(BaseSettings):
    """Google Gemini API configuration."""

    model_config = SettingsConfigDict(extra="ignore")

    api_key: Optional[str] = Field(
        default=None,
        alias="GOOGLE_API_KEY",
        description="Google API key",
    )

    @property
    def is_configured(self) -> bool:
        """Check if Gemini is properly configured."""
        return self.api_key is not None and len(self.api_key) > 0


class ServerSettings(BaseSettings):
    """API server configuration."""

    model_config = SettingsConfigDict(env_prefix="MICA_", extra="ignore")

    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    session_dir: Path = Field(
        default=Path("./sessions"),
        description="Session storage directory",
    )
    cors_origins: str = Field(
        default="http://localhost:3000,http://localhost:8080",
        description="Comma-separated CORS origins",
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key for authentication (optional)",
    )

    @property
    def cors_origins_list(self) -> list[str]:
        """Get CORS origins as a list."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


class SearchSettings(BaseSettings):
    """Web search configuration."""

    model_config = SettingsConfigDict(env_prefix="MICA_", extra="ignore")

    search_provider: Literal["tavily", "serpapi", "duckduckgo"] = Field(
        default="duckduckgo",
        description="Search provider to use",
    )


class TavilySettings(BaseSettings):
    """Tavily API configuration."""

    model_config = SettingsConfigDict(extra="ignore")

    api_key: Optional[str] = Field(
        default=None,
        alias="TAVILY_API_KEY",
        description="Tavily API key",
    )

    @property
    def is_configured(self) -> bool:
        return self.api_key is not None and len(self.api_key) > 0


class SerpAPISettings(BaseSettings):
    """SerpAPI configuration."""

    model_config = SettingsConfigDict(extra="ignore")

    api_key: Optional[str] = Field(
        default=None,
        alias="SERPAPI_API_KEY",
        description="SerpAPI key",
    )

    @property
    def is_configured(self) -> bool:
        return self.api_key is not None and len(self.api_key) > 0


class RAGSettings(BaseSettings):
    """RAG and ChromaDB configuration."""

    model_config = SettingsConfigDict(env_prefix="MICA_", extra="ignore")

    chroma_dir: Path = Field(
        default=Path("./chroma_db"),
        description="ChromaDB persistence directory",
    )
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence transformer model for embeddings",
    )


class LoggingSettings(BaseSettings):
    """Logging configuration."""

    model_config = SettingsConfigDict(env_prefix="MICA_", extra="ignore")

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Log level",
    )
    agent_logging: bool = Field(
        default=True,
        description="Enable detailed agent logging",
    )


class MICAConfig:
    """
    Main configuration container for MICA.

    Usage:
        from mica.config import config

        # Access LLM settings
        provider = config.llm.llm_provider
        model = config.llm.default_model

        # Check if Argo is configured
        if config.argo.is_configured:
            username = config.argo.username
    """

    def __init__(self):
        """Load all configuration from environment."""
        self._load_dotenv()

        self.llm = LLMSettings()
        self.argo = ArgoSettings()
        self.gemini = GeminiSettings()
        self.server = ServerSettings()
        self.search = SearchSettings()
        self.tavily = TavilySettings()
        self.serpapi = SerpAPISettings()
        self.rag = RAGSettings()
        self.logging = LoggingSettings()

    def _load_dotenv(self):
        """Load .env file if it exists."""
        try:
            from dotenv import load_dotenv

            # Look for .env in backend directory or project root
            backend_dir = Path(__file__).parent.parent
            project_root = backend_dir.parent

            for env_path in [backend_dir / ".env", project_root / ".env"]:
                if env_path.exists():
                    load_dotenv(env_path)
                    break
        except ImportError:
            pass  # python-dotenv not installed

    def get_active_llm_config(self) -> dict:
        """Get configuration for the active LLM provider."""
        if self.llm.llm_provider == "argo":
            if not self.argo.is_configured:
                raise ValueError(
                    "Argo LLM selected but ARGO_USERNAME not set. "
                    "Please set ARGO_USERNAME environment variable."
                )
            return {
                "provider": "argo",
                "model_id": self.llm.default_model,
                "username": self.argo.username,
            }
        elif self.llm.llm_provider == "gemini":
            if not self.gemini.is_configured:
                raise ValueError(
                    "Gemini LLM selected but GOOGLE_API_KEY not set. "
                    "Please set GOOGLE_API_KEY environment variable."
                )
            return {
                "provider": "gemini",
                "model_id": self.llm.default_model,
                "api_key": self.gemini.api_key,
            }
        else:
            raise ValueError(f"Unknown LLM provider: {self.llm.llm_provider}")

    def ensure_directories(self):
        """Create required directories if they don't exist."""
        self.server.session_dir.mkdir(parents=True, exist_ok=True)
        self.rag.chroma_dir.mkdir(parents=True, exist_ok=True)

    def validate(self) -> list[str]:
        """Validate configuration and return list of warnings/errors."""
        issues = []

        # Check LLM configuration
        if self.llm.llm_provider == "argo" and not self.argo.is_configured:
            issues.append("Argo selected but ARGO_USERNAME not set")
        if self.llm.llm_provider == "gemini" and not self.gemini.is_configured:
            issues.append("Gemini selected but GOOGLE_API_KEY not set")

        # Check search configuration
        if self.search.search_provider == "tavily" and not self.tavily.is_configured:
            issues.append("Tavily search selected but TAVILY_API_KEY not set")
        if self.search.search_provider == "serpapi" and not self.serpapi.is_configured:
            issues.append("SerpAPI search selected but SERPAPI_API_KEY not set")

        return issues


# Global configuration instance
config = MICAConfig()

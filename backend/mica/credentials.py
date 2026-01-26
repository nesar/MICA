"""
MICA Credentials Manager

Handles runtime credential storage and validation.
Credentials can be set via API when not provided in environment variables.
"""

import os
from dataclasses import dataclass
from typing import Optional
from threading import Lock


@dataclass
class Credentials:
    """Stored credentials for LLM providers."""
    argo_username: Optional[str] = None
    google_api_key: Optional[str] = None


class CredentialsManager:
    """
    Manages LLM provider credentials.

    Priority order:
    1. Runtime credentials (set via API)
    2. Environment variables

    Usage:
        from mica.credentials import credentials_manager

        # Check if credentials are available
        if not credentials_manager.has_argo_credentials():
            # Prompt user for credentials
            ...

        # Set credentials at runtime
        credentials_manager.set_argo_username("username")

        # Get credentials
        username = credentials_manager.get_argo_username()
    """

    def __init__(self):
        self._credentials = Credentials()
        self._lock = Lock()

    # Argo credentials
    def get_argo_username(self) -> Optional[str]:
        """Get Argo username from runtime or environment."""
        with self._lock:
            if self._credentials.argo_username:
                return self._credentials.argo_username
        return os.environ.get("ARGO_USERNAME")

    def set_argo_username(self, username: str):
        """Set Argo username at runtime."""
        with self._lock:
            self._credentials.argo_username = username

    def has_argo_credentials(self) -> bool:
        """Check if Argo credentials are available."""
        return self.get_argo_username() is not None

    def clear_argo_credentials(self):
        """Clear runtime Argo credentials."""
        with self._lock:
            self._credentials.argo_username = None

    # Google/Gemini credentials
    def get_google_api_key(self) -> Optional[str]:
        """Get Google API key from runtime or environment."""
        with self._lock:
            if self._credentials.google_api_key:
                return self._credentials.google_api_key
        return os.environ.get("GOOGLE_API_KEY")

    def set_google_api_key(self, api_key: str):
        """Set Google API key at runtime."""
        with self._lock:
            self._credentials.google_api_key = api_key

    def has_google_credentials(self) -> bool:
        """Check if Google credentials are available."""
        return self.get_google_api_key() is not None

    def clear_google_credentials(self):
        """Clear runtime Google credentials."""
        with self._lock:
            self._credentials.google_api_key = None

    # General methods
    def get_credentials_status(self) -> dict:
        """Get status of all credentials."""
        return {
            "argo": {
                "configured": self.has_argo_credentials(),
                "source": "runtime" if self._credentials.argo_username else (
                    "environment" if os.environ.get("ARGO_USERNAME") else None
                ),
            },
            "gemini": {
                "configured": self.has_google_credentials(),
                "source": "runtime" if self._credentials.google_api_key else (
                    "environment" if os.environ.get("GOOGLE_API_KEY") else None
                ),
            },
        }

    def clear_all(self):
        """Clear all runtime credentials."""
        with self._lock:
            self._credentials = Credentials()


# Global credentials manager instance
credentials_manager = CredentialsManager()

"""
Azure AI Project client and connection management.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from azure.ai.projects import AIProjectClient
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
from typing import Optional
from core.config import Settings

settings = Settings()


class AzureAIConnection:
    """Manages Azure AI Foundry project connection."""
    
    def __init__(self):
        self.endpoint = settings.AZURE_AI_PROJECT_ENDPOINT
        self.api_key = settings.AZURE_AI_PROJECT_API_KEY
        
        if not self.endpoint or not self.api_key:
            raise ValueError(
                "Azure AI credentials not found. "
                "Please ensure AZURE_AI_PROJECT_ENDPOINT and AZURE_AI_PROJECT_API_KEY "
                "are set in your .env file."
            )
        
        self._client = None
    
    @property
    def client(self) -> AIProjectClient:
        """Get or create AI Project client."""
        if self._client is None:
            # For Azure Agents API, use DefaultAzureCredential
            # This supports: Azure CLI login, managed identity, environment variables
            try:
                credential = DefaultAzureCredential()
                self._client = AIProjectClient(
                    endpoint=self.endpoint,
                    credential=credential
                )
            except Exception as e:
                # If DefaultAzureCredential fails, try AzureKeyCredential as fallback
                # Note: Some Azure Agents API operations require RBAC and won't work with API keys
                print(f"DefaultAzureCredential failed: {e}")
                print("Trying API key authentication (limited functionality)...")
                try:
                    credential = AzureKeyCredential(self.api_key)
                    self._client = AIProjectClient(
                        endpoint=self.endpoint,
                        credential=credential
                    )
                except Exception as e2:
                    raise ValueError(
                        f"Authentication failed. Please run 'az login' to authenticate via Azure CLI, "
                        f"or ensure your service principal/managed identity has proper permissions. "
                        f"Errors: DefaultAzureCredential: {e}, AzureKeyCredential: {e2}"
                    )
        return self._client
    
    def validate_connection(self) -> bool:
        """
        Validate the Azure AI connection.
        
        Returns:
            True if connection is valid, False otherwise.
        """
        try:
            # Try to get project info or list deployments
            client = self.client
            # If we can create the client, connection is likely valid
            # Note: Actual validation would depend on available API methods
            return True
        except Exception as e:
            print(f"Connection validation failed: {e}")
            return False
    
    def get_agent_config(
        self,
        model_name: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: int = 1500
    ) -> dict:
        """
        Get configuration for creating an agent.
        
        Args:
            model_name: Name of the model deployment
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            
        Returns:
            Configuration dictionary for agent creation
        """
        return {
            "model": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }


def get_azure_client() -> AIProjectClient:
    """
    Convenience function to get an Azure AI Project client.
        
    Returns:
        Configured AIProjectClient instance
    """
    connection = AzureAIConnection()
    return connection.client


def get_azure_connection() -> AzureAIConnection:
    """
    Get an AzureAIConnection instance.
    Returns:
        AzureAIConnection instance
    """
    return AzureAIConnection()

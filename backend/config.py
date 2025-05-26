import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, Field, ValidationError
from platformdirs import user_config_dir, user_log_dir

logger = logging.getLogger(__name__)

APP_NAME = "LightChat"
APP_AUTHOR = "SudoSynthesis.dev"


class ProviderConfig(BaseModel):
    """Configuration for an LLM provider."""
    id: str = Field(
        ...,
        description="A unique identifier for this provider configuration"
    )
    name: str = Field(
        ...,
        description="A user-friendly display name for this provider"
    )
    type: Literal["ollama", "openai"] = Field(
        ...,
        description="The type of the provider"
    )
    api_key: Optional[str] = Field(
        None,
        description="API key for the provider (required for OpenAI)"
    )
    host: Optional[str] = Field(
        None,
        description="Host URL for the provider (required for Ollama)"
    )
    system_prompt: Optional[str] = Field(
        None,
        description="Default system prompt for this provider"
    )
    default_model: Optional[str] = Field(
        None,
        description="Default model to use if not specified in the request"
    )

class AppConfig(BaseModel):
    """Application configuration model.
    
    This model defines the structure and default values for the application's
    configuration. It uses Pydantic for data validation and serialization.
    """
    default_provider: Optional[str] = Field(
        default=None,
        description="Default LLM provider ID. This ID is used to identify a specific provider configuration."
    )
    
    providers: List[ProviderConfig] = Field(
        default_factory=list,
        description="List of configured LLM providers"
    )
    
    log_dir: str = Field(
        default_factory=lambda: str(Path(user_log_dir(APP_NAME, APP_AUTHOR))),
        description="Directory where log files will be stored."
    )
    
    data_dir: str = Field(
        default_factory=lambda: str(Path(user_config_dir(APP_NAME, APP_AUTHOR)) / "data"),
        description="Directory where application data will be stored."
    )
    
    logging_enabled: bool = Field(
        default=True,
        description="Whether logging is enabled for the application."
    )
    
    debug: bool = Field(
        default=False,
        description="Whether debug mode is enabled for the application."
    )
    
    def get_provider(self, provider_id: str) -> Optional[ProviderConfig]:
        """Get a provider configuration by ID.
        
        Args:
            provider_id: The ID of the provider to get
            
        Returns:
            ProviderConfig or None if not found
        """
        for provider in self.providers:
            if provider.id == provider_id:
                return provider
        return None


def get_config_path() -> Path:
    """Get the path to the configuration file.
    
    Returns:
        Path: The path to the settings.yaml file.
    """
    config_dir_path = Path(user_config_dir(APP_NAME, APP_AUTHOR))
    return config_dir_path / "settings.yaml"


def load_app_config() -> AppConfig:
    """Load the application configuration from the settings file.
    
    If the settings file doesn't exist, contains invalid YAML, or fails validation,
    a new AppConfig instance with default values is returned.
    
    Returns:
        AppConfig: The loaded or default application configuration.
    """
    config_path = get_config_path()
    
    if not config_path.exists():
        logger.info(f"Configuration file not found at {config_path}, using default settings.")
        return AppConfig()
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f) or {}
        
        return AppConfig(**config_data)
        
    except yaml.YAMLError as e:
        logger.warning(
            f"Failed to parse configuration file at {config_path}: {e}\n"
            "Using default settings."
        )
        return AppConfig()
    except ValidationError as e:
        logger.warning(
            f"Configuration validation error in {config_path}:\n{e}\n"
            "Using default settings."
        )
        return AppConfig()
    except Exception as e:
        logger.warning(
            f"Unexpected error loading configuration from {config_path}: {e}\n"
            "Using default settings.",
            exc_info=True
        )
        return AppConfig()


def save_app_config(config: AppConfig) -> None:
    """Save the application configuration to the settings file.
    
    Args:
        config: The AppConfig instance to save.
        
    Raises:
        IOError: If the configuration directory cannot be created or the file cannot be written.
    """
    config_path = get_config_path()
    config_dir = config_path.parent
    
    try:
        config_dir.mkdir(parents=True, exist_ok=True)
        
        config_dict = config.model_dump(exclude_unset=True, exclude_none=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
            
        logger.info(f"Configuration saved to {config_path}")
        
    except OSError as e:
        logger.critical(
            f"Failed to save configuration to {config_path}: {e}"
        )
        raise IOError(f"Failed to save configuration: {e}") from e
"""Backend package for the LightChat application."""

from .config import (
    AppConfig,
    load_app_config,
    save_app_config,
    get_config_path,
    APP_NAME,
    APP_AUTHOR,
)

__all__ = [
    'AppConfig',
    'load_app_config',
    'save_app_config',
    'get_config_path',
    'APP_NAME',
    'APP_AUTHOR',
]

"""Unit tests for the config module."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock
import pytest
import yaml

from backend.config import (
    AppConfig,
    load_app_config,
    save_app_config,
    get_config_path,
    APP_NAME,
    APP_AUTHOR,
)

@patch('backend.config.user_log_dir')
def test_app_config_defaults(mock_user_log_dir):
    """Test that AppConfig has the correct default values."""
    # Setup mock to return a test path
    test_log_dir = "/tmp/mock/log/dir"
    mock_user_log_dir.return_value = test_log_dir
    
    config = AppConfig()
    assert config.default_provider is None
    assert config.log_dir == test_log_dir
    assert config.logging_enabled is True
    mock_user_log_dir.assert_called_once_with(APP_NAME, APP_AUTHOR)


def test_app_config_override():
    """Test that AppConfig can be initialized with custom values."""
    config = AppConfig(
        default_provider="test_provider",
        log_dir="/custom/log/dir",
        logging_enabled=False
    )
    assert config.default_provider == "test_provider"
    assert config.log_dir == "/custom/log/dir"
    assert config.logging_enabled is False


@patch('backend.config.get_config_path')
def test_load_app_config_missing_file(mock_get_path):
    """Test loading config when the file doesn't exist."""
    mock_path_obj = MagicMock(spec=Path)
    mock_path_obj.exists.return_value = False
    mock_path_obj.__str__.return_value = "/mock/config/LightChat/settings.yaml"
    mock_get_path.return_value = mock_path_obj

    with patch('logging.Logger.info') as mock_info:
        config = load_app_config()
        assert isinstance(config, AppConfig)
        assert config.default_provider is AppConfig.model_fields['default_provider'].default
        assert config.logging_enabled is AppConfig.model_fields['logging_enabled'].default
        mock_info.assert_called_once()
        assert "Configuration file not found" in mock_info.call_args[0][0]


@patch('backend.config.get_config_path')
def test_load_app_config_valid_file(mock_get_path):
    """Test loading config from a valid YAML file."""
    mock_path_obj = MagicMock(spec=Path)
    mock_path_obj.exists.return_value = True
    mock_get_path.return_value = mock_path_obj

    config_data = {
        "default_provider": "test_provider",
        "log_dir": "/custom/log/dir",
        "logging_enabled": False
    }
    yaml_content = yaml.dump(config_data)
    
    with patch('builtins.open', mock_open(read_data=yaml_content)):
        config = load_app_config()
        assert config.default_provider == "test_provider"
        assert config.log_dir == "/custom/log/dir"
        assert "data" in config.data_dir  # Just check that it's a valid path
        assert config.logging_enabled is False
        assert config.debug is False


@patch('backend.config.get_config_path')
def test_load_app_config_empty_yaml_file(mock_get_path):
    """Test loading config from an existing but empty YAML file."""
    mock_path_obj = MagicMock(spec=Path)
    mock_path_obj.exists.return_value = True
    mock_get_path.return_value = mock_path_obj
    
    with patch('builtins.open', mock_open(read_data="")), \
         patch('logging.Logger.warning') as mock_warning:
        config = load_app_config()
        assert isinstance(config, AppConfig)
        assert config.default_provider is AppConfig.model_fields['default_provider'].default
        assert config.log_dir
        assert config.data_dir
        assert config.logging_enabled is AppConfig.model_fields['logging_enabled'].default
        assert config.debug is AppConfig.model_fields['debug'].default
        mock_warning.assert_not_called()


@patch('backend.config.get_config_path')
def test_load_app_config_invalid_yaml(mock_get_path):
    """Test loading config from a file with invalid YAML."""
    mock_path_obj = MagicMock(spec=Path)
    mock_path_obj.exists.return_value = True
    mock_path_obj.__str__.return_value = "/mock/config/invalid.yaml"
    mock_get_path.return_value = mock_path_obj
    
    with patch('builtins.open', mock_open(read_data='invalid: yaml: : : :')), \
         patch('logging.Logger.warning') as mock_warning:
        config = load_app_config()
        assert isinstance(config, AppConfig)
        mock_warning.assert_called_once()
        assert "Failed to parse configuration file" in mock_warning.call_args[0][0]
        assert "/mock/config/invalid.yaml" in mock_warning.call_args[0][0]


@patch('backend.config.get_config_path')
def test_load_app_config_invalid_schema(mock_get_path):
    """Test loading config with invalid schema (wrong data types)."""
    mock_path_obj = MagicMock(spec=Path)
    mock_path_obj.exists.return_value = True
    mock_path_obj.__str__.return_value = "/mock/config/invalid_schema.yaml"
    mock_get_path.return_value = mock_path_obj

    invalid_data = {
        "default_provider": 123,
        "logging_enabled": "not a boolean"
    }
    yaml_content = yaml.dump(invalid_data)
    
    with patch('builtins.open', mock_open(read_data=yaml_content)), \
         patch('logging.Logger.warning') as mock_warning:
        config = load_app_config()
        assert isinstance(config, AppConfig)
        mock_warning.assert_called_once()
        log_message = mock_warning.call_args[0][0]
        assert "Configuration validation error" in log_message
        assert "/mock/config/invalid_schema.yaml" in log_message
        assert "default_provider" in log_message
        assert "logging_enabled" in log_message


@patch('backend.config.get_config_path')
def test_save_app_config(mock_get_path):
    """Test saving the configuration to a file."""
    mock_config_path = MagicMock(spec=Path)
    mock_config_dir = MagicMock(spec=Path)
    
    mock_get_path.return_value = mock_config_path
    mock_config_path.parent = mock_config_dir
    mock_config_path.__str__.return_value = "/mock/config/dir/settings.yaml"

    config = AppConfig(
        default_provider="test_provider",
        log_dir="/custom/log/dir",
        logging_enabled=False
    )
    
    mock_file_open = mock_open()
    with patch('builtins.open', mock_file_open), \
         patch('logging.Logger.info') as mock_log_info:
        
        save_app_config(config)
        
        mock_config_dir.mkdir.assert_called_once_with(parents=True, exist_ok=True)
        mock_file_open.assert_called_once_with(mock_config_path, 'w', encoding='utf-8')
        
        handle = mock_file_open()
        handle.write.assert_called()
        
        written_content = "".join(
            call[0][0] for call in handle.write.call_args_list
        )
        
        saved_data = yaml.safe_load(written_content)
        assert saved_data["default_provider"] == "test_provider"
        assert saved_data["log_dir"] == "/custom/log/dir"
        assert saved_data["logging_enabled"] is False
        mock_log_info.assert_called_with("Configuration saved to /mock/config/dir/settings.yaml")


@patch('backend.config.get_config_path')
def test_save_app_config_io_error(mock_get_path):
    """Test handling of IOError during config save."""
    mock_config_path = MagicMock(spec=Path)
    mock_config_dir = MagicMock(spec=Path)
    mock_get_path.return_value = mock_config_path
    mock_config_path.parent = mock_config_dir
    mock_config_path.__str__.return_value = "/mock/config/io_error.yaml"

    # Make directory creation succeed but file open fail
    mock_config_dir.mkdir.return_value = None 
    
    config = AppConfig()
    
    with patch('builtins.open', side_effect=IOError("Permission denied")), \
         patch('logging.Logger.critical') as mock_critical, \
         pytest.raises(IOError, match="Failed to save configuration: Permission denied"):
        
        save_app_config(config)
        mock_critical.assert_called_once()
        assert "/mock/config/io_error.yaml" in mock_critical.call_args[0][0]


@patch('backend.config.user_config_dir')
def test_get_config_path(mock_user_config_dir):
    """Test that the config path is constructed correctly."""
    # Setup mock to return a test path
    test_config_dir = "/tmp/mock/config/dir"
    mock_user_config_dir.return_value = test_config_dir
    
    config_path = get_config_path()
    
    # Verify the function was called with the correct arguments
    mock_user_config_dir.assert_called_once_with(APP_NAME, APP_AUTHOR)
    # Verify the path is constructed correctly
    assert str(config_path) == f"{test_config_dir}/settings.yaml"

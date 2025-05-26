import pytest
import logging
import json
import os
import time
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, call

from backend.logger import JSONFormatter, setup_logging

# Mock AppConfig for tests
class MockAppConfig:
    def __init__(self, log_dir, logging_enabled=True):
        self.log_dir = str(log_dir)  # Ensure log_dir is a string for Path operations
        self.logging_enabled = logging_enabled

@pytest.fixture
def json_formatter_instance():
    return JSONFormatter()

@pytest.fixture
def mock_app_config_fixture(tmp_path: Path):
    # tmp_path is a pytest fixture providing a temporary directory unique to the test invocation
    return MockAppConfig(log_dir=tmp_path)

@pytest.fixture
def sample_log_record():
    return logging.LogRecord(
        name='test_logger',
        level=logging.INFO,
        pathname='test_module.py',
        lineno=10,
        msg='This is a test message: %s',
        args=('some_arg',),
        exc_info=None,
        func='test_function'
    )

# --- JSONFormatter Tests --- 

def test_json_formatter_basic_fields(json_formatter_instance: JSONFormatter, sample_log_record: logging.LogRecord):
    formatted_log = json_formatter_instance.format(sample_log_record)
    log_data = json.loads(formatted_log)

    assert log_data['level'] == 'INFO'
    assert log_data['message'] == 'This is a test message: some_arg'
    assert log_data['module'] == 'test_module'
    assert log_data['function'] == 'test_function'
    assert log_data['lineno'] == 10
    assert 'timestamp' in log_data
    assert 'threadName' in log_data
    assert 'process' in log_data

    # Verify timestamp is ISO format and UTC
    parsed_timestamp = datetime.fromisoformat(log_data['timestamp'].replace('Z', '+00:00'))
    assert parsed_timestamp.tzinfo == timezone.utc

def test_json_formatter_with_extra_fields(json_formatter_instance: JSONFormatter, sample_log_record: logging.LogRecord):
    sample_log_record.extra = {'custom_field': 'custom_value', 'request_id': 123}
    # Simulate how logging module adds 'extra' to the record dictionary for our formatter
    sample_log_record.__dict__.update(sample_log_record.extra)
    
    formatted_log = json_formatter_instance.format(sample_log_record)
    log_data = json.loads(formatted_log)

    assert log_data['custom_field'] == 'custom_value'
    assert log_data['request_id'] == 123
    assert log_data['message'] == 'This is a test message: some_arg'

# --- setup_logging Tests --- 

# Fixture to manage logger state across tests
@pytest.fixture(autouse=True)
def isolated_logger():
    # This fixture will run for every test function
    # Setup: Store original handlers and level if any
    app_logger = logging.getLogger("lightchat")
    original_handlers = list(app_logger.handlers)
    original_level = app_logger.level
    original_propagate = app_logger.propagate
    
    # Ensure logger is clean before test
    for handler in original_handlers:
        app_logger.removeHandler(handler)
        if hasattr(handler, 'close'): 
            handler.close()
    app_logger.setLevel(logging.NOTSET)  # Reset level
    app_logger.propagate = True  # Reset propagate

    yield  # Test runs here

    # Teardown: Clean up
    current_handlers = list(app_logger.handlers)
    for handler in current_handlers:
        app_logger.removeHandler(handler)
        if hasattr(handler, 'close'): 
            handler.close()
    
    # Reset to known state
    app_logger.setLevel(logging.NOTSET)
    app_logger.propagate = True

def test_setup_logging_directory_and_file_creation(mock_app_config_fixture: MockAppConfig, isolated_logger):
    listener = None
    try:
        listener = setup_logging(mock_app_config_fixture)
        logger = logging.getLogger("lightchat")  # Get the configured logger
        
        log_base_dir = Path(mock_app_config_fixture.log_dir)
        expected_log_dir = log_base_dir / "LightChat" / "logs"
        expected_log_file = expected_log_dir / "lightchat.log"

        assert expected_log_dir.exists(), "Log directory was not created"
        assert expected_log_dir.is_dir(), "Log path is not a directory"
        
        test_message = "Test log message for file creation."
        logger.info(test_message, extra={'test_key': 'test_value'})
        
        time.sleep(0.2)  # Allow time for async processing

        assert expected_log_file.exists(), "Log file was not created"
        assert expected_log_file.is_file(), "Log path is not a file"

        with open(expected_log_file, 'r') as f:
            lines = f.readlines()
            assert len(lines) >= 2, "Expected at least two log lines (init + test message)"
            
            # Check initialization message
            init_log_json = json.loads(lines[0])
            assert init_log_json["message"] == "Logging system initialized."
            assert init_log_json["setup_phase"] == "logging"

            # Check test message
            test_log_json = json.loads(lines[1])
            assert test_log_json["message"] == test_message
            assert test_log_json["test_key"] == "test_value"

    finally:
        if listener:
            listener.stop()

@patch('logging.handlers.RotatingFileHandler')
@patch('logging.handlers.QueueHandler')
@patch('logging.handlers.QueueListener')
@patch('logging.getLogger')
def test_setup_logging_handler_configuration(
    mock_get_logger: MagicMock,
    MockQueueListener: MagicMock,
    MockQueueHandler: MagicMock,
    MockRotatingFileHandler: MagicMock,
    mock_app_config_fixture: MockAppConfig,
    isolated_logger
):
    # Create a mock logger to return from logging.getLogger
    mock_logger = MagicMock()
    
    # Track the logger's level
    mock_logger.level = logging.NOTSET
    def set_level(level):
        mock_logger.level = level
    mock_logger.setLevel.side_effect = set_level
    
    # Track the propagate flag
    mock_logger.propagate = True
    
    # Use a real list that we can track for handlers
    mock_handlers = []
    
    # Configure the handlers property to return our tracked list
    type(mock_logger).handlers = mock_handlers
    
    # Set up the addHandler method to append to our list
    def add_handler(handler):
        mock_handlers.append(handler)
    mock_logger.addHandler.side_effect = add_handler
    
    # Configure the mock to return our logger
    mock_get_logger.return_value = mock_logger
    
    # Create a mock handler with a proper level attribute
    mock_handler = MagicMock()
    mock_handler.level = logging.INFO
    MockRotatingFileHandler.return_value = mock_handler
    
    # Create a mock queue handler that will be returned by QueueHandler()
    mock_queue_handler = MagicMock()
    MockQueueHandler.return_value = mock_queue_handler
    
    # Mock the formatter
    mock_formatter = MagicMock(spec=JSONFormatter)
    with patch('backend.logger.JSONFormatter', return_value=mock_formatter) as MockJSONFormatter:
        listener_instance = setup_logging(mock_app_config_fixture)

        # Check RotatingFileHandler instantiation
        log_dir = Path(mock_app_config_fixture.log_dir) / "LightChat" / "logs"
        active_log_file = log_dir / "lightchat.log"
        
        MockRotatingFileHandler.assert_called_once_with(
            filename=active_log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        
        # Check formatter was set on the handler
        MockRotatingFileHandler.return_value.setFormatter.assert_called_once_with(mock_formatter)

        # Check QueueHandler was created with a queue
        MockQueueHandler.assert_called_once()
        log_queue = MockQueueHandler.call_args[0][0]
        assert hasattr(log_queue, 'put')  # Basic check that it's a queue-like object

        # Check QueueListener was set up correctly
        MockQueueListener.assert_called_once_with(
            log_queue,
            MockRotatingFileHandler.return_value,
            respect_handler_level=True
        )
        MockQueueListener.return_value.start.assert_called_once()
        
        # Check logger configuration
        mock_get_logger.assert_called_once_with("lightchat")
        
        # Verify the handler was added to the logger
        mock_logger.addHandler.assert_called_once_with(mock_queue_handler)
        
        # Verify the logger's properties were set correctly
        assert mock_logger.level == logging.INFO
        assert not mock_logger.propagate
        
        # Verify the handler is in the handlers list
        assert len(mock_handlers) == 1
        assert mock_handlers[0] == mock_queue_handler

        # Clean up
        if listener_instance:
            listener_instance.stop()

@pytest.mark.parametrize(
    "level, message, extra, expected_level_str",
    [
        (logging.DEBUG, "Debug message", {'user': 'tester'}, "DEBUG"),
        (logging.INFO, "Info message", None, "INFO"),
        (logging.WARNING, "Warning message", None, "WARNING"),
        (logging.ERROR, "Error message", {'code': 404}, "ERROR"),
        (logging.CRITICAL, "Critical error", {'code': 500}, "CRITICAL"),
    ]
)
def test_json_formatter_levels_and_extra(json_formatter_instance: JSONFormatter, level, message, extra, expected_level_str):
    record = logging.LogRecord(
        name='param_test', level=level, pathname='param.py', lineno=1, 
        msg=message, args=(), exc_info=None, func='param_func'
    )
    if extra:
        # Set the extra fields on the record
        record.extra = extra
        # Also update __dict__ to simulate both ways extra fields can be set
        for k, v in extra.items():
            setattr(record, k, v)
    
    formatted = json_formatter_instance.format(record)
    log_data = json.loads(formatted)
    
    # Check standard fields
    assert log_data['level'] == expected_level_str
    assert log_data['message'] == message
    assert log_data['module'] == 'param'
    assert log_data['function'] == 'param_func'
    assert log_data['lineno'] == 1
    
    # Check extra fields are included at the top level
    if extra:
        for k, v in extra.items():
            assert k in log_data, f"Extra field '{k}' not found in log entry"
            assert log_data[k] == v, f"Value mismatch for field '{k}': expected {v}, got {log_data[k]}"

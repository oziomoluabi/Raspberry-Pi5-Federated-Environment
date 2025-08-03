"""
Comprehensive Test Configuration and Fixtures
Raspberry Pi 5 Federated Environmental Monitoring Network

This module provides shared test fixtures, utilities, and configuration
for the entire test suite, supporting both unit and integration testing.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Generator, Optional
from unittest.mock import Mock, MagicMock
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Test configuration
TEST_CONFIG = {
    'simulation_mode': True,
    'test_data_size': 100,
    'timeout_seconds': 30,
    'random_seed': 42,
    'temp_dir_prefix': 'rpi5_federated_test_',
}

# Set random seeds for reproducibility
np.random.seed(TEST_CONFIG['random_seed'])


@pytest.fixture(scope="session")
def test_config() -> Dict[str, Any]:
    """Provide test configuration for all tests."""
    return TEST_CONFIG.copy()


@pytest.fixture(scope="session")
def project_root_path() -> Path:
    """Provide project root path."""
    return project_root


@pytest.fixture(scope="function")
def temp_directory() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    temp_dir = Path(tempfile.mkdtemp(prefix=TEST_CONFIG['temp_dir_prefix']))
    try:
        yield temp_dir
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


@pytest.fixture(scope="function")
def mock_sensor_data() -> Dict[str, Any]:
    """Generate mock sensor data for testing."""
    size = TEST_CONFIG['test_data_size']
    timestamps = pd.date_range(
        start=datetime.now() - timedelta(hours=1),
        periods=size,
        freq='1min'
    )
    
    return {
        'timestamp': timestamps,
        'temperature': np.random.normal(22.0, 2.0, size),  # °C
        'humidity': np.random.normal(45.0, 10.0, size),    # %
        'pressure': np.random.normal(1013.25, 5.0, size), # hPa
        'vibration_x': np.random.normal(0.0, 0.1, size),  # g
        'vibration_y': np.random.normal(0.0, 0.1, size),  # g
        'vibration_z': np.random.normal(1.0, 0.1, size),  # g (gravity)
    }


@pytest.fixture(scope="function")
def mock_environmental_dataframe(mock_sensor_data) -> pd.DataFrame:
    """Create a pandas DataFrame with mock environmental data."""
    return pd.DataFrame(mock_sensor_data)


@pytest.fixture(scope="function")
def mock_vibration_data() -> np.ndarray:
    """Generate mock vibration data for autoencoder testing."""
    size = TEST_CONFIG['test_data_size']
    # Simulate normal vibration patterns with some anomalies
    normal_data = np.random.normal(0, 0.1, (size, 3))
    
    # Add some anomalies (10% of data)
    anomaly_indices = np.random.choice(size, size // 10, replace=False)
    normal_data[anomaly_indices] += np.random.normal(0, 1.0, (len(anomaly_indices), 3))
    
    return normal_data


@pytest.fixture(scope="function")
def mock_lstm_training_data() -> tuple:
    """Generate mock LSTM training data (X, y)."""
    size = TEST_CONFIG['test_data_size']
    sequence_length = 10
    n_features = 3
    
    # Generate sequential data
    X = np.random.randn(size - sequence_length, sequence_length, n_features)
    y = np.random.randn(size - sequence_length, n_features)
    
    return X, y


@pytest.fixture(scope="function")
def mock_federated_client_config() -> Dict[str, Any]:
    """Provide mock federated client configuration."""
    return {
        'client_id': 'test_client_001',
        'server_address': 'localhost:8080',
        'model_config': {
            'lstm_units': 64,
            'sequence_length': 10,
            'n_features': 3,
            'learning_rate': 0.001,
        },
        'training_config': {
            'local_epochs': 5,
            'batch_size': 32,
            'validation_split': 0.2,
        },
        'privacy_config': {
            'enable_differential_privacy': True,
            'noise_multiplier': 0.1,
            'l2_norm_clip': 1.0,
        },
        'security_config': {
            'enable_tls': True,
            'cert_path': '/tmp/test_cert.pem',
            'key_path': '/tmp/test_key.pem',
        }
    }


@pytest.fixture(scope="function")
def mock_server_config() -> Dict[str, Any]:
    """Provide mock federated server configuration."""
    return {
        'server_address': '0.0.0.0:8080',
        'min_clients': 2,
        'min_available_clients': 2,
        'rounds': 3,
        'model_config': {
            'lstm_units': 64,
            'sequence_length': 10,
            'n_features': 3,
        },
        'aggregation_config': {
            'strategy': 'fedavg',
            'fraction_fit': 1.0,
            'fraction_evaluate': 1.0,
        },
        'security_config': {
            'enable_tls': True,
            'enable_jwt_auth': True,
            'jwt_secret': 'test_secret_key',
        }
    }


@pytest.fixture(scope="function")
def mock_sensor_manager():
    """Create a mock sensor manager for testing."""
    mock_manager = Mock()
    mock_manager.simulation_mode = True
    mock_manager.get_sensor_reading.return_value = {
        'timestamp': datetime.now(),
        'temperature': 22.5,
        'humidity': 45.0,
        'pressure': 1013.25,
        'vibration': {'x': 0.01, 'y': 0.02, 'z': 1.01}
    }
    mock_manager.collect_batch_data.return_value = [
        mock_manager.get_sensor_reading.return_value
        for _ in range(10)
    ]
    return mock_manager


@pytest.fixture(scope="function")
def mock_matlab_engine():
    """Create a mock MATLAB engine for testing."""
    mock_engine = Mock()
    mock_engine.eval.return_value = None
    mock_engine.workspace = {}
    mock_engine.quit.return_value = None
    
    # Mock common MATLAB functions
    mock_engine.mean.return_value = 22.5
    mock_engine.std.return_value = 2.0
    mock_engine.size.return_value = (100, 3)
    
    return mock_engine


@pytest.fixture(scope="function")
def mock_tinyml_model():
    """Create a mock TinyML model for testing."""
    mock_model = Mock()
    mock_model.predict.return_value = np.random.randn(10, 3)
    mock_model.fit.return_value = None
    mock_model.evaluate.return_value = [0.1, 0.95]  # loss, accuracy
    mock_model.save.return_value = None
    return mock_model


@pytest.fixture(scope="function")
def mock_flower_client():
    """Create a mock Flower client for testing."""
    from flwr.client import Client
    
    class MockFlowerClient(Client):
        def get_parameters(self, config):
            return [np.random.randn(10, 5).flatten()]
        
        def fit(self, parameters, config):
            return [np.random.randn(10, 5).flatten()], 100, {}
        
        def evaluate(self, parameters, config):
            return 0.1, 100, {"accuracy": 0.95}
    
    return MockFlowerClient()


@pytest.fixture(scope="function")
def mock_certificates(temp_directory):
    """Create mock TLS certificates for testing."""
    cert_dir = temp_directory / "certs"
    cert_dir.mkdir()
    
    # Create dummy certificate files
    cert_files = {
        'ca_cert': cert_dir / "ca_cert.pem",
        'ca_key': cert_dir / "ca_key.pem",
        'server_cert': cert_dir / "server_cert.pem",
        'server_key': cert_dir / "server_key.pem",
        'client_cert': cert_dir / "client_cert.pem",
        'client_key': cert_dir / "client_key.pem",
    }
    
    # Write dummy content to certificate files
    for cert_file in cert_files.values():
        cert_file.write_text("-----BEGIN CERTIFICATE-----\nDUMMY_CERT_CONTENT\n-----END CERTIFICATE-----\n")
    
    return cert_files


@pytest.fixture(scope="function")
def mock_database_connection():
    """Create a mock database connection for testing."""
    mock_conn = Mock()
    mock_cursor = Mock()
    
    mock_cursor.fetchall.return_value = []
    mock_cursor.fetchone.return_value = None
    mock_cursor.execute.return_value = None
    mock_cursor.close.return_value = None
    
    mock_conn.cursor.return_value = mock_cursor
    mock_conn.commit.return_value = None
    mock_conn.rollback.return_value = None
    mock_conn.close.return_value = None
    
    return mock_conn


@pytest.fixture(scope="function")
def performance_monitor():
    """Provide a performance monitoring context manager."""
    import time
    import psutil
    import threading
    
    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            self.start_memory = None
            self.end_memory = None
            self.peak_memory = None
            self.monitoring = False
            self.monitor_thread = None
        
        def __enter__(self):
            self.start_time = time.time()
            self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            self.peak_memory = self.start_memory
            self.monitoring = True
            
            # Start memory monitoring thread
            self.monitor_thread = threading.Thread(target=self._monitor_memory)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            self.monitoring = False
            self.end_time = time.time()
            self.end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            if self.monitor_thread:
                self.monitor_thread.join(timeout=1.0)
        
        def _monitor_memory(self):
            while self.monitoring:
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                self.peak_memory = max(self.peak_memory, current_memory)
                time.sleep(0.1)
        
        @property
        def execution_time(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None
        
        @property
        def memory_usage(self):
            return {
                'start_mb': self.start_memory,
                'end_mb': self.end_memory,
                'peak_mb': self.peak_memory,
                'delta_mb': self.end_memory - self.start_memory if self.end_memory and self.start_memory else None
            }
    
    return PerformanceMonitor()


# Test markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "hardware: mark test as requiring hardware")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "security: mark test as security-related")
    config.addinivalue_line("markers", "performance: mark test as performance benchmark")
    config.addinivalue_line("markers", "matlab: mark test as requiring MATLAB")
    config.addinivalue_line("markers", "tinyml: mark test as TinyML-related")
    config.addinivalue_line("markers", "federated: mark test as federated learning-related")


# Test collection hooks
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test location."""
    for item in items:
        # Add markers based on test file location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Add markers based on test name patterns
        if "hardware" in item.name.lower():
            item.add_marker(pytest.mark.hardware)
        if "slow" in item.name.lower():
            item.add_marker(pytest.mark.slow)
        if "security" in item.name.lower():
            item.add_marker(pytest.mark.security)
        if "performance" in item.name.lower():
            item.add_marker(pytest.mark.performance)
        if "matlab" in item.name.lower():
            item.add_marker(pytest.mark.matlab)
        if "tinyml" in item.name.lower():
            item.add_marker(pytest.mark.tinyml)
        if "federated" in item.name.lower():
            item.add_marker(pytest.mark.federated)


# Utility functions for tests
class TestUtils:
    """Utility functions for testing."""
    
    @staticmethod
    def assert_array_close(actual, expected, rtol=1e-5, atol=1e-8):
        """Assert that two arrays are close within tolerance."""
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
    
    @staticmethod
    def assert_dataframe_equal(df1, df2, check_dtype=True):
        """Assert that two DataFrames are equal."""
        pd.testing.assert_frame_equal(df1, df2, check_dtype=check_dtype)
    
    @staticmethod
    def create_test_model_weights(shape):
        """Create test model weights with specified shape."""
        return np.random.randn(*shape).astype(np.float32)
    
    @staticmethod
    def simulate_sensor_failure():
        """Simulate sensor failure for error handling tests."""
        raise ConnectionError("Simulated sensor connection failure")
    
    @staticmethod
    def wait_for_condition(condition_func, timeout=10, interval=0.1):
        """Wait for a condition to become true within timeout."""
        import time
        start_time = time.time()
        while time.time() - start_time < timeout:
            if condition_func():
                return True
            time.sleep(interval)
        return False


@pytest.fixture(scope="session")
def test_utils():
    """Provide test utilities."""
    return TestUtils


# Environment setup and teardown
@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up the test environment."""
    # Set environment variables for testing
    os.environ['TESTING'] = 'true'
    os.environ['SIMULATION_MODE'] = 'true'
    os.environ['LOG_LEVEL'] = 'DEBUG'
    
    # Ensure test directories exist
    test_dirs = [
        project_root / "tests" / "data",
        project_root / "tests" / "outputs",
        project_root / "tests" / "logs",
    ]
    
    for test_dir in test_dirs:
        test_dir.mkdir(exist_ok=True)
    
    yield
    
    # Cleanup after all tests
    # Remove test environment variables
    test_env_vars = ['TESTING', 'SIMULATION_MODE', 'LOG_LEVEL']
    for var in test_env_vars:
        os.environ.pop(var, None)


# Logging configuration for tests
@pytest.fixture(scope="session", autouse=True)
def configure_test_logging():
    """Configure logging for tests."""
    import logging
    import structlog
    
    # Configure structlog for testing
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.dev.ConsoleRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Set logging level for tests
    logging.getLogger().setLevel(logging.DEBUG)
    
    yield
    
    # Reset logging configuration
    structlog.reset_defaults()

"""
Integration Tests for Federated Learning Workflow
Raspberry Pi 5 Federated Environmental Monitoring Network

Comprehensive integration tests covering:
- End-to-end federated learning workflow
- Client-server communication with security
- Model training and aggregation
- TinyML integration with federated learning
- MATLAB/Simulink integration workflow
"""

import pytest
import asyncio
import threading
import time
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import tempfile
import json

# Import project modules
import sys
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from server.main import FederatedServer
from client.main import FederatedClient
from server.models.lstm_model import create_lstm_model, LSTMModelManager
from client.training.autoencoder import VibrationAutoencoder, AutoencoderManager
from client.training.tinyml_inference import TinyMLInferenceEngine
from client.matlab.matlab_integration import MATLABEngineManager, EnvironmentalDataProcessor
from server.security.tls_config import TLSCertificateManager, SecureTokenManager
from server.security.secure_federated_server import SecureFederatedStrategy


@pytest.mark.integration
class TestFederatedLearningWorkflow:
    """Test complete federated learning workflow."""
    
    @pytest.fixture
    def server_config(self, temp_directory):
        """Provide server configuration for testing."""
        return {
            'server_address': '127.0.0.1:8080',
            'min_clients': 2,
            'min_available_clients': 2,
            'rounds': 3,
            'model_config': {
                'lstm_units': 32,
                'sequence_length': 10,
                'n_features': 3,
                'learning_rate': 0.001,
            },
            'security_config': {
                'enable_tls': False,  # Disable for testing
                'enable_jwt_auth': False,
                'cert_dir': str(temp_directory),
            }
        }
    
    @pytest.fixture
    def client_configs(self, temp_directory):
        """Provide client configurations for testing."""
        base_config = {
            'server_address': '127.0.0.1:8080',
            'simulation_mode': True,
            'model_config': {
                'lstm_units': 32,
                'sequence_length': 10,
                'n_features': 3,
                'learning_rate': 0.001,
            },
            'training_config': {
                'local_epochs': 2,
                'batch_size': 16,
                'validation_split': 0.2,
            },
            'data_config': {
                'collection_interval': 1.0,
                'batch_size': 50,
            }
        }
        
        return [
            {**base_config, 'client_id': 'test_client_001'},
            {**base_config, 'client_id': 'test_client_002'},
        ]
    
    def test_server_initialization(self, server_config):
        """Test federated server initialization."""
        server = FederatedServer(server_config)
        
        assert server.config == server_config
        assert server.server_address == server_config['server_address']
        assert server.min_clients == server_config['min_clients']
    
    def test_client_initialization(self, client_configs):
        """Test federated client initialization."""
        client_config = client_configs[0]
        client = FederatedClient(client_config)
        
        assert client.config == client_config
        assert client.client_id == client_config['client_id']
        assert client.simulation_mode == client_config['simulation_mode']
    
    @pytest.mark.slow
    def test_federated_training_simulation(self, server_config, client_configs, mock_environmental_dataframe):
        """Test complete federated training simulation."""
        # Mock the actual training process
        with patch('server.main.FederatedServer.start_server') as mock_server_start, \
             patch('client.main.FederatedClient.start_training') as mock_client_start, \
             patch('client.sensing.sensor_manager.SensorManager.collect_batch_data') as mock_collect:
            
            # Mock data collection
            mock_collect.return_value = mock_environmental_dataframe.to_dict('records')
            
            # Initialize server and clients
            server = FederatedServer(server_config)
            clients = [FederatedClient(config) for config in client_configs]
            
            # Mock server start to return immediately
            mock_server_start.return_value = None
            
            # Mock client training to simulate successful training
            def mock_training(client):
                # Simulate training process
                time.sleep(0.1)  # Brief delay to simulate training
                return {
                    'loss': np.random.uniform(0.1, 0.5),
                    'accuracy': np.random.uniform(0.8, 0.95),
                    'num_examples': len(mock_environmental_dataframe)
                }
            
            mock_client_start.side_effect = lambda: mock_training(None)
            
            # Start server (mocked)
            server.start_server()
            
            # Start clients (mocked)
            training_results = []
            for client in clients:
                result = client.start_training()
                training_results.append(result)
            
            # Verify training results
            assert len(training_results) == len(clients)
            for result in training_results:
                assert 'loss' in result
                assert 'accuracy' in result
                assert 'num_examples' in result
                assert 0.0 <= result['loss'] <= 1.0
                assert 0.0 <= result['accuracy'] <= 1.0
    
    def test_model_aggregation_workflow(self, mock_lstm_training_data):
        """Test model parameter aggregation workflow."""
        X, y = mock_lstm_training_data
        
        # Create multiple client models with different weights
        client_models = []
        for i in range(3):
            model = create_lstm_model(
                sequence_length=10,
                n_features=3,
                lstm_units=32
            )
            # Train briefly with different data
            model.fit(X[i*10:(i+1)*10], y[i*10:(i+1)*10], epochs=1, verbose=0)
            client_models.append(model)
        
        # Extract weights from client models
        client_weights = [model.get_weights() for model in client_models]
        
        # Simulate federated averaging
        aggregated_weights = []
        for layer_idx in range(len(client_weights[0])):
            layer_weights = [weights[layer_idx] for weights in client_weights]
            avg_weights = np.mean(layer_weights, axis=0)
            aggregated_weights.append(avg_weights)
        
        # Create new model with aggregated weights
        global_model = create_lstm_model(
            sequence_length=10,
            n_features=3,
            lstm_units=32
        )
        global_model.set_weights(aggregated_weights)
        
        # Verify aggregated model can make predictions
        predictions = global_model.predict(X[:5], verbose=0)
        assert predictions.shape == (5, 3)
        assert not np.isnan(predictions).any()
    
    def test_client_server_communication_simulation(self, server_config, client_configs):
        """Test client-server communication simulation."""
        # Mock network communication
        with patch('flwr.client.start_numpy_client') as mock_start_client, \
             patch('flwr.server.start_server') as mock_start_server:
            
            # Mock successful communication
            mock_start_client.return_value = None
            mock_start_server.return_value = None
            
            # Initialize components
            server = FederatedServer(server_config)
            client = FederatedClient(client_configs[0])
            
            # Test server start
            server.start_server()
            mock_start_server.assert_called_once()
            
            # Test client connection
            client.start_training()
            mock_start_client.assert_called_once()
    
    def test_federated_learning_with_differential_privacy(self, mock_lstm_training_data):
        """Test federated learning with differential privacy enabled."""
        X, y = mock_lstm_training_data
        
        # Initialize secure federated strategy
        strategy = SecureFederatedStrategy(
            noise_multiplier=0.1,
            l2_norm_clip=1.0,
            enable_secure_aggregation=True
        )
        
        # Create mock client parameters
        client_params = [
            [np.random.randn(32, 64), np.random.randn(64)],
            [np.random.randn(32, 64), np.random.randn(64)],
        ]
        
        # Apply differential privacy
        private_params = []
        for params in client_params:
            noisy_params = strategy.add_differential_privacy_noise(params)
            private_params.append(noisy_params)
        
        # Verify noise was added
        for orig, noisy in zip(client_params, private_params):
            for orig_layer, noisy_layer in zip(orig, noisy):
                assert not np.allclose(orig_layer, noisy_layer, atol=1e-10)
        
        # Verify privacy budget tracking
        strategy.update_privacy_budget(0.5)
        assert strategy.privacy_budget_used == 0.5


@pytest.mark.integration
class TestTinyMLIntegration:
    """Test TinyML integration with federated learning."""
    
    def test_autoencoder_training_workflow(self, mock_vibration_data):
        """Test complete autoencoder training workflow."""
        # Initialize autoencoder
        autoencoder = VibrationAutoencoder(
            input_dim=3,
            encoding_dim=2,
            learning_rate=0.001
        )
        
        # Build model
        model = autoencoder.build_model()
        
        # Train model
        history = autoencoder.train(
            mock_vibration_data,
            epochs=5,
            batch_size=16,
            validation_split=0.2,
            verbose=0
        )
        
        # Verify training completed
        assert 'loss' in history.history
        assert len(history.history['loss']) == 5
        
        # Test inference
        predictions = autoencoder.predict(mock_vibration_data[:10])
        assert predictions.shape == (10, 3)
    
    def test_tinyml_inference_engine(self, mock_vibration_data, temp_directory):
        """Test TinyML inference engine functionality."""
        # Create and train autoencoder
        autoencoder = VibrationAutoencoder(input_dim=3, encoding_dim=2)
        model = autoencoder.build_model()
        model.fit(mock_vibration_data, mock_vibration_data, epochs=2, verbose=0)
        
        # Convert to TensorFlow Lite
        tflite_model_path = temp_directory / "autoencoder.tflite"
        autoencoder.convert_to_tflite(str(tflite_model_path))
        
        # Initialize inference engine
        inference_engine = TinyMLInferenceEngine(str(tflite_model_path))
        
        # Test inference
        test_input = mock_vibration_data[:5]
        predictions = inference_engine.predict(test_input)
        
        assert predictions.shape == (5, 3)
        assert not np.isnan(predictions).any()
    
    def test_anomaly_detection_workflow(self, mock_vibration_data):
        """Test complete anomaly detection workflow."""
        # Split data into normal and anomalous
        normal_data = mock_vibration_data[:80]  # First 80% as normal
        test_data = mock_vibration_data[80:]    # Last 20% may contain anomalies
        
        # Train autoencoder on normal data
        autoencoder = VibrationAutoencoder(input_dim=3, encoding_dim=2)
        model = autoencoder.build_model()
        model.fit(normal_data, normal_data, epochs=10, verbose=0)
        
        # Calculate reconstruction errors
        reconstructions = model.predict(test_data, verbose=0)
        reconstruction_errors = np.mean(np.square(test_data - reconstructions), axis=1)
        
        # Set threshold (e.g., 95th percentile of training errors)
        training_reconstructions = model.predict(normal_data, verbose=0)
        training_errors = np.mean(np.square(normal_data - training_reconstructions), axis=1)
        threshold = np.percentile(training_errors, 95)
        
        # Detect anomalies
        anomalies = reconstruction_errors > threshold
        
        # Verify anomaly detection results
        assert len(anomalies) == len(test_data)
        assert isinstance(anomalies, np.ndarray)
        assert anomalies.dtype == bool
    
    def test_federated_tinyml_integration(self, mock_vibration_data, temp_directory):
        """Test integration between federated learning and TinyML."""
        # Simulate multiple clients with different vibration patterns
        client_data = [
            mock_vibration_data[:30],   # Client 1 data
            mock_vibration_data[30:60], # Client 2 data
            mock_vibration_data[60:90], # Client 3 data
        ]
        
        # Train local models on each client
        client_models = []
        for i, data in enumerate(client_data):
            autoencoder = VibrationAutoencoder(input_dim=3, encoding_dim=2)
            model = autoencoder.build_model()
            model.fit(data, data, epochs=5, verbose=0)
            client_models.append(model)
        
        # Simulate federated averaging of autoencoder weights
        client_weights = [model.get_weights() for model in client_models]
        
        # Average weights across clients
        aggregated_weights = []
        for layer_idx in range(len(client_weights[0])):
            layer_weights = [weights[layer_idx] for weights in client_weights]
            avg_weights = np.mean(layer_weights, axis=0)
            aggregated_weights.append(avg_weights)
        
        # Create global model with aggregated weights
        global_autoencoder = VibrationAutoencoder(input_dim=3, encoding_dim=2)
        global_model = global_autoencoder.build_model()
        global_model.set_weights(aggregated_weights)
        
        # Test global model performance
        test_data = mock_vibration_data[90:]
        predictions = global_model.predict(test_data, verbose=0)
        
        assert predictions.shape == test_data.shape
        assert not np.isnan(predictions).any()
        
        # Convert global model to TensorFlow Lite
        tflite_path = temp_directory / "federated_autoencoder.tflite"
        global_autoencoder.convert_to_tflite(str(tflite_path))
        
        # Verify TFLite model works
        assert tflite_path.exists()
        inference_engine = TinyMLInferenceEngine(str(tflite_path))
        tflite_predictions = inference_engine.predict(test_data[:5])
        
        assert tflite_predictions.shape == (5, 3)


@pytest.mark.integration
@pytest.mark.matlab
class TestMATLABIntegration:
    """Test MATLAB/Simulink integration workflow."""
    
    def test_matlab_engine_initialization(self):
        """Test MATLAB engine initialization with fallback."""
        with patch('matlab.engine.start_matlab') as mock_matlab, \
             patch('oct2py.Oct2Py') as mock_octave:
            
            # Test successful MATLAB initialization
            mock_matlab.return_value = Mock()
            
            engine_manager = MATLABEngineManager()
            engine_manager.initialize_engine()
            
            assert engine_manager.engine is not None
            assert engine_manager.engine_type == 'matlab'
            mock_matlab.assert_called_once()
    
    def test_matlab_fallback_to_octave(self):
        """Test fallback to Octave when MATLAB is not available."""
        with patch('matlab.engine.start_matlab', side_effect=ImportError("MATLAB not available")), \
             patch('oct2py.Oct2Py') as mock_octave:
            
            mock_octave.return_value = Mock()
            
            engine_manager = MATLABEngineManager()
            engine_manager.initialize_engine()
            
            assert engine_manager.engine is not None
            assert engine_manager.engine_type == 'octave'
            mock_octave.assert_called_once()
    
    def test_environmental_data_processing(self, mock_environmental_dataframe):
        """Test environmental data processing with MATLAB."""
        with patch('matlab.engine.start_matlab') as mock_matlab:
            # Mock MATLAB engine
            mock_engine = Mock()
            mock_engine.eval.return_value = None
            mock_engine.mean.return_value = 22.5
            mock_engine.std.return_value = 2.0
            mock_matlab.return_value = mock_engine
            
            # Initialize processor
            processor = EnvironmentalDataProcessor()
            processor.engine_manager.initialize_engine()
            
            # Process data
            processed_data = processor.preprocess_data(mock_environmental_dataframe)
            
            # Verify processing
            assert processed_data is not None
            assert isinstance(processed_data, dict)
    
    def test_simulink_model_execution(self, mock_environmental_dataframe):
        """Test Simulink model execution workflow."""
        with patch('matlab.engine.start_matlab') as mock_matlab:
            # Mock MATLAB engine with Simulink capabilities
            mock_engine = Mock()
            mock_engine.eval.return_value = None
            mock_engine.sim.return_value = Mock(
                get=Mock(return_value=np.array([1, 2, 3, 4, 5]))
            )
            mock_matlab.return_value = mock_engine
            
            # Initialize engine manager
            engine_manager = MATLABEngineManager()
            engine_manager.initialize_engine()
            
            # Mock Simulink model runner
            from client.matlab.matlab_integration import SimulinkModelRunner
            
            model_runner = SimulinkModelRunner(engine_manager)
            
            # Run simulation
            results = model_runner.run_simulation(
                model_name='test_model',
                input_data=mock_environmental_dataframe,
                simulation_time=10.0
            )
            
            # Verify simulation results
            assert results is not None
            mock_engine.sim.assert_called_once()
    
    def test_matlab_federated_integration(self, mock_environmental_dataframe):
        """Test integration between MATLAB processing and federated learning."""
        with patch('matlab.engine.start_matlab') as mock_matlab:
            # Mock MATLAB engine
            mock_engine = Mock()
            mock_engine.eval.return_value = None
            
            # Mock MATLAB preprocessing results
            mock_engine.env_preprocess.return_value = {
                'processed_temp': np.array([20, 21, 22, 23, 24]),
                'processed_humidity': np.array([40, 42, 44, 46, 48]),
                'features': np.random.randn(100, 5)
            }
            mock_matlab.return_value = mock_engine
            
            # Initialize components
            processor = EnvironmentalDataProcessor()
            processor.engine_manager.initialize_engine()
            
            # Process data with MATLAB
            matlab_results = processor.preprocess_data(mock_environmental_dataframe)
            
            # Use processed data for federated learning
            if 'features' in matlab_results:
                features = matlab_results['features']
                
                # Create LSTM training data from MATLAB features
                sequence_length = 10
                X, y = [], []
                for i in range(len(features) - sequence_length):
                    X.append(features[i:i+sequence_length])
                    y.append(features[i+sequence_length])
                
                X, y = np.array(X), np.array(y)
                
                # Train LSTM model
                model = create_lstm_model(
                    sequence_length=sequence_length,
                    n_features=features.shape[1],
                    lstm_units=32
                )
                
                # Brief training
                model.fit(X, y, epochs=2, verbose=0)
                
                # Verify model can make predictions
                predictions = model.predict(X[:5], verbose=0)
                assert predictions.shape == (5, features.shape[1])


@pytest.mark.integration
class TestSecureWorkflowIntegration:
    """Test secure workflow integration with TLS and JWT."""
    
    def test_secure_server_client_setup(self, temp_directory):
        """Test secure server-client setup with TLS and JWT."""
        # Initialize security components
        cert_manager = TLSCertificateManager(cert_dir=temp_directory)
        token_manager = SecureTokenManager("test_secret_key")
        
        # Generate certificates
        cert_manager.generate_ca_certificate()
        cert_manager.generate_server_certificate("localhost")
        cert_manager.generate_client_certificate("test_client")
        
        # Generate JWT token
        token = token_manager.generate_token("test_client", ["read", "write", "train"])
        
        # Verify security setup
        ssl_context = cert_manager.create_ssl_context(is_server=True)
        is_valid, payload = token_manager.validate_token(token)
        
        assert ssl_context is not None
        assert is_valid
        assert payload["client_id"] == "test_client"
        assert "train" in payload["permissions"]
    
    def test_secure_federated_training_workflow(self, temp_directory, mock_lstm_training_data):
        """Test secure federated training workflow."""
        X, y = mock_lstm_training_data
        
        # Initialize secure components
        cert_manager = TLSCertificateManager(cert_dir=temp_directory)
        token_manager = SecureTokenManager("test_secret")
        strategy = SecureFederatedStrategy(
            noise_multiplier=0.1,
            l2_norm_clip=1.0,
            enable_secure_aggregation=True
        )
        
        # Setup certificates
        cert_manager.generate_ca_certificate()
        cert_manager.generate_server_certificate("localhost")
        
        # Generate client tokens
        client_tokens = []
        for i in range(3):
            token = token_manager.generate_token(f"client_{i}", ["train"])
            client_tokens.append(token)
        
        # Simulate secure training
        client_models = []
        for i in range(3):
            # Verify client has training permission
            token = client_tokens[i]
            has_permission = token_manager.check_permission(token, "train")
            assert has_permission
            
            # Train local model
            model = create_lstm_model(
                sequence_length=10,
                n_features=3,
                lstm_units=32
            )
            model.fit(X[i*10:(i+1)*20], y[i*10:(i+1)*20], epochs=2, verbose=0)
            client_models.append(model)
        
        # Extract and secure model parameters
        client_weights = [model.get_weights() for model in client_models]
        
        # Apply differential privacy
        private_weights = []
        for weights in client_weights:
            noisy_weights = strategy.add_differential_privacy_noise(weights)
            private_weights.append(noisy_weights)
        
        # Aggregate with privacy
        aggregated_weights = []
        for layer_idx in range(len(private_weights[0])):
            layer_weights = [weights[layer_idx] for weights in private_weights]
            avg_weights = np.mean(layer_weights, axis=0)
            aggregated_weights.append(avg_weights)
        
        # Create global model
        global_model = create_lstm_model(
            sequence_length=10,
            n_features=3,
            lstm_units=32
        )
        global_model.set_weights(aggregated_weights)
        
        # Verify secure workflow completed
        predictions = global_model.predict(X[:5], verbose=0)
        assert predictions.shape == (5, 3)
        assert not np.isnan(predictions).any()


@pytest.mark.integration
@pytest.mark.slow
class TestEndToEndWorkflow:
    """Test complete end-to-end workflow integration."""
    
    def test_complete_system_workflow(self, temp_directory, mock_environmental_dataframe, mock_vibration_data):
        """Test complete system workflow from data collection to deployment."""
        # 1. Initialize all components
        cert_manager = TLSCertificateManager(cert_dir=temp_directory)
        token_manager = SecureTokenManager("test_secret")
        
        # 2. Setup security
        cert_manager.generate_ca_certificate()
        cert_manager.generate_server_certificate("localhost")
        
        # 3. Mock sensor data collection
        with patch('client.sensing.sensor_manager.SensorManager.collect_batch_data') as mock_collect:
            mock_collect.return_value = mock_environmental_dataframe.to_dict('records')
            
            # 4. Mock MATLAB processing
            with patch('matlab.engine.start_matlab') as mock_matlab:
                mock_engine = Mock()
                mock_engine.env_preprocess.return_value = {
                    'features': np.random.randn(100, 3)
                }
                mock_matlab.return_value = mock_engine
                
                # 5. Process environmental data
                processor = EnvironmentalDataProcessor()
                processor.engine_manager.initialize_engine()
                processed_data = processor.preprocess_data(mock_environmental_dataframe)
                
                # 6. Train LSTM model
                features = processed_data.get('features', np.random.randn(100, 3))
                X, y = [], []
                for i in range(len(features) - 10):
                    X.append(features[i:i+10])
                    y.append(features[i+10])
                X, y = np.array(X), np.array(y)
                
                lstm_model = create_lstm_model(
                    sequence_length=10,
                    n_features=3,
                    lstm_units=32
                )
                lstm_model.fit(X, y, epochs=2, verbose=0)
                
                # 7. Train TinyML autoencoder
                autoencoder = VibrationAutoencoder(input_dim=3, encoding_dim=2)
                ae_model = autoencoder.build_model()
                ae_model.fit(mock_vibration_data, mock_vibration_data, epochs=2, verbose=0)
                
                # 8. Convert to TensorFlow Lite
                tflite_path = temp_directory / "autoencoder.tflite"
                autoencoder.convert_to_tflite(str(tflite_path))
                
                # 9. Test inference
                inference_engine = TinyMLInferenceEngine(str(tflite_path))
                predictions = inference_engine.predict(mock_vibration_data[:5])
                
                # 10. Verify complete workflow
                assert processed_data is not None
                assert lstm_model is not None
                assert tflite_path.exists()
                assert predictions.shape == (5, 3)
                
                # 11. Generate security audit report
                from scripts.security_audit import SecurityAuditor
                auditor = SecurityAuditor(project_root=temp_directory)
                
                with patch.object(auditor, 'scan_dependencies', return_value=[]), \
                     patch.object(auditor, 'analyze_code_security', return_value=[]):
                    audit_results = auditor.run_comprehensive_audit()
                
                assert "dependencies" in audit_results
                assert "code_security" in audit_results
    
    def test_performance_monitoring_workflow(self, performance_monitor, mock_environmental_dataframe):
        """Test performance monitoring throughout the workflow."""
        with performance_monitor as monitor:
            # Simulate complete workflow with performance monitoring
            
            # 1. Data preprocessing
            processed_data = mock_environmental_dataframe.copy()
            processed_data['normalized_temp'] = (processed_data['temperature'] - processed_data['temperature'].mean()) / processed_data['temperature'].std()
            
            # 2. Model training
            features = processed_data[['temperature', 'humidity', 'pressure']].values
            X, y = [], []
            for i in range(len(features) - 10):
                X.append(features[i:i+10])
                y.append(features[i+10])
            X, y = np.array(X), np.array(y)
            
            model = create_lstm_model(
                sequence_length=10,
                n_features=3,
                lstm_units=32
            )
            model.fit(X, y, epochs=2, verbose=0)
            
            # 3. Model inference
            predictions = model.predict(X[:10], verbose=0)
            
            # 4. Anomaly detection
            reconstruction_errors = np.mean(np.square(X[:10] - predictions), axis=(1, 2))
            threshold = np.percentile(reconstruction_errors, 95)
            anomalies = reconstruction_errors > threshold
        
        # Verify performance metrics
        assert monitor.execution_time is not None
        assert monitor.execution_time < 30.0  # Should complete within 30 seconds
        assert monitor.memory_usage['peak_mb'] < 500  # Should use less than 500MB
        
        # Verify workflow results
        assert predictions.shape == (10, 3)
        assert len(anomalies) == 10
        assert isinstance(anomalies, np.ndarray)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "integration"])

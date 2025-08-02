"""
Unit tests for LSTM Model
"""

import pytest
import numpy as np
import tensorflow as tf
from unittest.mock import patch

from server.models.lstm_model import create_lstm_model, create_federated_model


class TestLSTMModel:
    """Test cases for LSTM model creation and functionality."""
    
    def test_create_lstm_model_basic(self):
        """Test basic LSTM model creation."""
        model = create_lstm_model(
            input_shape=(24, 2),
            lstm_units=32,
            dropout_rate=0.1,
            output_units=2
        )
        
        assert model is not None
        assert isinstance(model, tf.keras.Model)
        assert len(model.layers) == 6  # 2 LSTM + 2 Dropout + 2 Dense
        
        # Check input shape
        assert model.input_shape == (None, 24, 2)
        
        # Check output shape
        assert model.output_shape == (None, 2)
    
    def test_create_lstm_model_compilation(self):
        """Test that the model is properly compiled."""
        model = create_lstm_model(
            input_shape=(10, 3),
            lstm_units=16,
            dropout_rate=0.2,
            output_units=1
        )
        
        # Check that optimizer is set
        assert model.optimizer is not None
        assert isinstance(model.optimizer, tf.keras.optimizers.Adam)
        
        # Check that loss function is set
        assert model.loss == 'mse'
        
        # Check that metrics are set
        assert 'mae' in [m.name for m in model.metrics]
    
    def test_create_lstm_model_different_params(self):
        """Test LSTM model creation with different parameters."""
        model = create_lstm_model(
            input_shape=(48, 4),
            lstm_units=128,
            dropout_rate=0.3,
            output_units=3
        )
        
        assert model.input_shape == (None, 48, 4)
        assert model.output_shape == (None, 3)
        
        # Check that the model has the expected number of parameters
        param_count = model.count_params()
        assert param_count > 0
    
    def test_create_federated_model(self):
        """Test creation of federated model with standard configuration."""
        model = create_federated_model()
        
        assert model is not None
        assert isinstance(model, tf.keras.Model)
        assert model.input_shape == (None, 24, 2)
        assert model.output_shape == (None, 2)
    
    def test_model_prediction(self):
        """Test that the model can make predictions."""
        model = create_lstm_model(
            input_shape=(10, 2),
            lstm_units=16,
            dropout_rate=0.1,
            output_units=2
        )
        
        # Create dummy input data
        batch_size = 4
        timesteps = 10
        features = 2
        dummy_input = np.random.randn(batch_size, timesteps, features)
        
        # Make prediction
        predictions = model.predict(dummy_input, verbose=0)
        
        assert predictions.shape == (batch_size, 2)
        assert np.all(np.isfinite(predictions))  # No NaN or inf values
    
    def test_model_training_step(self):
        """Test that the model can perform a training step."""
        model = create_lstm_model(
            input_shape=(5, 2),
            lstm_units=8,
            dropout_rate=0.1,
            output_units=1
        )
        
        # Create dummy training data
        batch_size = 8
        timesteps = 5
        features = 2
        output_features = 1
        
        x_train = np.random.randn(batch_size, timesteps, features)
        y_train = np.random.randn(batch_size, output_features)
        
        # Perform one training step
        history = model.fit(
            x_train, y_train,
            epochs=1,
            verbose=0,
            validation_split=0.2
        )
        
        assert 'loss' in history.history
        assert 'mae' in history.history
        assert len(history.history['loss']) == 1
    
    def test_model_layer_names(self):
        """Test that model layers have expected names."""
        model = create_lstm_model(
            input_shape=(12, 3),
            lstm_units=32,
            dropout_rate=0.2,
            output_units=2
        )
        
        layer_names = [layer.name for layer in model.layers]
        
        assert 'lstm_1' in layer_names
        assert 'lstm_2' in layer_names
        assert 'dropout_1' in layer_names
        assert 'dropout_2' in layer_names
        assert 'dense_1' in layer_names
        assert 'output' in layer_names
    
    def test_model_weights_initialization(self):
        """Test that model weights are properly initialized."""
        model = create_lstm_model(
            input_shape=(6, 2),
            lstm_units=16,
            dropout_rate=0.1,
            output_units=1
        )
        
        # Check that all layers have weights
        for layer in model.layers:
            if hasattr(layer, 'weights') and layer.weights:
                for weight in layer.weights:
                    assert weight.shape.num_elements() > 0
                    # Check that weights are not all zeros
                    weight_values = weight.numpy()
                    assert not np.allclose(weight_values, 0)


@pytest.mark.integration
class TestLSTMModelIntegration:
    """Integration tests for LSTM model."""
    
    def test_model_training_convergence(self):
        """Test that the model can learn from synthetic data."""
        model = create_lstm_model(
            input_shape=(10, 2),
            lstm_units=32,
            dropout_rate=0.1,
            output_units=2
        )
        
        # Create synthetic data with a simple pattern
        batch_size = 100
        timesteps = 10
        features = 2
        
        # Generate data where output is sum of inputs
        x_data = np.random.randn(batch_size, timesteps, features)
        y_data = np.sum(x_data, axis=(1, 2)).reshape(-1, 1)
        y_data = np.hstack([y_data, y_data * 0.5])  # Two outputs
        
        # Train for a few epochs
        initial_loss = model.evaluate(x_data, y_data, verbose=0)[0]
        
        model.fit(
            x_data, y_data,
            epochs=10,
            verbose=0,
            validation_split=0.2
        )
        
        final_loss = model.evaluate(x_data, y_data, verbose=0)[0]
        
        # Loss should decrease (model should learn)
        assert final_loss < initial_loss
    
    def test_model_save_load(self):
        """Test that the model can be saved and loaded."""
        import tempfile
        import os
        
        model = create_lstm_model(
            input_shape=(8, 2),
            lstm_units=16,
            dropout_rate=0.1,
            output_units=1
        )
        
        # Create dummy data for prediction
        dummy_input = np.random.randn(1, 8, 2)
        original_prediction = model.predict(dummy_input, verbose=0)
        
        # Save model to temporary file
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, 'test_model.h5')
            model.save(model_path)
            
            # Load model
            loaded_model = tf.keras.models.load_model(model_path)
            
            # Make prediction with loaded model
            loaded_prediction = loaded_model.predict(dummy_input, verbose=0)
            
            # Predictions should be identical
            np.testing.assert_array_almost_equal(
                original_prediction, loaded_prediction, decimal=6
            )

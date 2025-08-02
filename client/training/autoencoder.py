"""
TinyML Autoencoder for Vibration Anomaly Detection
Raspberry Pi 5 Federated Environmental Monitoring Network
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Tuple, Optional, Dict, List
import structlog
from pathlib import Path
import json
import time

logger = structlog.get_logger(__name__)


class VibrationAutoencoder:
    """Autoencoder for vibration anomaly detection with TinyML optimization."""
    
    def __init__(
        self,
        input_dim: int = 128,
        encoding_dim: int = 32,
        learning_rate: float = 0.001,
        model_name: str = "vibration_autoencoder"
    ):
        """Initialize the autoencoder."""
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.learning_rate = learning_rate
        self.model_name = model_name
        self.model = None
        self.encoder = None
        self.decoder = None
        self.threshold = None
        
        logger.info(
            "Initializing vibration autoencoder",
            input_dim=input_dim,
            encoding_dim=encoding_dim,
            learning_rate=learning_rate
        )
    
    def build_model(self) -> keras.Model:
        """Build the autoencoder model optimized for TinyML."""
        
        # Input layer
        input_layer = keras.Input(shape=(self.input_dim,), name="input")
        
        # Encoder layers (progressively smaller for TinyML efficiency)
        encoded = layers.Dense(64, activation="relu", name="encoder_1")(input_layer)
        encoded = layers.Dropout(0.1, name="encoder_dropout_1")(encoded)
        encoded = layers.Dense(self.encoding_dim, activation="relu", name="encoder_2")(encoded)
        
        # Decoder layers (mirror of encoder)
        decoded = layers.Dense(64, activation="relu", name="decoder_1")(encoded)
        decoded = layers.Dropout(0.1, name="decoder_dropout_1")(decoded)
        decoded = layers.Dense(self.input_dim, activation="linear", name="decoder_2")(decoded)
        
        # Create models
        self.model = keras.Model(input_layer, decoded, name=self.model_name)
        self.encoder = keras.Model(input_layer, encoded, name=f"{self.model_name}_encoder")
        
        # Compile with MSE loss for reconstruction
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="mse",
            metrics=["mae"]
        )
        
        logger.info(
            "Autoencoder model built",
            total_params=self.model.count_params(),
            encoder_params=self.encoder.count_params()
        )
        
        return self.model
    
    def generate_synthetic_vibration_data(
        self,
        num_samples: int = 1000,
        anomaly_ratio: float = 0.1,
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic vibration data with anomalies."""
        
        if seed is not None:
            np.random.seed(seed)
        
        logger.info(
            "Generating synthetic vibration data",
            samples=num_samples,
            anomaly_ratio=anomaly_ratio,
            input_dim=self.input_dim
        )
        
        # Normal vibration patterns (low frequency, low amplitude)
        normal_samples = int(num_samples * (1 - anomaly_ratio))
        
        # Generate normal vibration data
        t = np.linspace(0, 1, self.input_dim)
        normal_data = []
        
        for i in range(normal_samples):
            # Base frequency components (normal operation)
            freq1 = np.random.uniform(5, 15)  # Primary frequency
            freq2 = np.random.uniform(20, 40)  # Secondary harmonic
            
            # Normal amplitude
            amp1 = np.random.uniform(0.1, 0.3)
            amp2 = np.random.uniform(0.05, 0.15)
            
            # Generate signal
            signal = (
                amp1 * np.sin(2 * np.pi * freq1 * t) +
                amp2 * np.sin(2 * np.pi * freq2 * t) +
                np.random.normal(0, 0.02, self.input_dim)  # Low noise
            )
            
            normal_data.append(signal)
        
        # Generate anomalous vibration data
        anomaly_samples = num_samples - normal_samples
        anomaly_data = []
        
        for i in range(anomaly_samples):
            # Anomalous patterns (high frequency, high amplitude, or irregular)
            anomaly_type = np.random.choice(['high_freq', 'high_amp', 'irregular'])
            
            if anomaly_type == 'high_freq':
                # High frequency vibrations (bearing issues)
                freq1 = np.random.uniform(80, 150)
                freq2 = np.random.uniform(200, 300)
                amp1 = np.random.uniform(0.2, 0.5)
                amp2 = np.random.uniform(0.1, 0.3)
                
            elif anomaly_type == 'high_amp':
                # High amplitude vibrations (imbalance)
                freq1 = np.random.uniform(10, 25)
                freq2 = np.random.uniform(30, 50)
                amp1 = np.random.uniform(0.8, 1.5)
                amp2 = np.random.uniform(0.4, 0.8)
                
            else:  # irregular
                # Irregular patterns (looseness, misalignment)
                freq1 = np.random.uniform(5, 50)
                freq2 = np.random.uniform(60, 120)
                amp1 = np.random.uniform(0.3, 0.8)
                amp2 = np.random.uniform(0.2, 0.6)
            
            # Generate anomalous signal
            signal = (
                amp1 * np.sin(2 * np.pi * freq1 * t) +
                amp2 * np.sin(2 * np.pi * freq2 * t) +
                np.random.normal(0, 0.1, self.input_dim)  # Higher noise
            )
            
            anomaly_data.append(signal)
        
        # Combine and shuffle data
        all_data = np.array(normal_data + anomaly_data)
        labels = np.array([0] * normal_samples + [1] * anomaly_samples)
        
        # Shuffle
        indices = np.random.permutation(num_samples)
        all_data = all_data[indices]
        labels = labels[indices]
        
        logger.info(
            "Synthetic vibration data generated",
            total_samples=num_samples,
            normal_samples=normal_samples,
            anomaly_samples=anomaly_samples,
            data_shape=all_data.shape
        )
        
        return all_data, labels
    
    def train(
        self,
        train_data: np.ndarray,
        validation_data: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 32,
        verbose: int = 1
    ) -> Dict:
        """Train the autoencoder on normal vibration data."""
        
        if self.model is None:
            self.build_model()
        
        logger.info(
            "Starting autoencoder training",
            train_samples=len(train_data),
            epochs=epochs,
            batch_size=batch_size
        )
        
        # Prepare validation data
        validation_data_tuple = None
        if validation_data is not None:
            validation_data_tuple = (validation_data, validation_data)
        
        # Train the model (autoencoder learns to reconstruct input)
        start_time = time.time()
        history = self.model.fit(
            train_data, train_data,  # Input and target are the same
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data_tuple,
            verbose=verbose,
            shuffle=True
        )
        
        training_time = time.time() - start_time
        
        # Calculate reconstruction threshold on training data
        train_predictions = self.model.predict(train_data, verbose=0)
        train_mse = np.mean(np.square(train_data - train_predictions), axis=1)
        self.threshold = np.percentile(train_mse, 95)  # 95th percentile as threshold
        
        logger.info(
            "Autoencoder training completed",
            training_time=f"{training_time:.2f}s",
            final_loss=history.history['loss'][-1],
            threshold=self.threshold
        )
        
        return {
            'history': history.history,
            'training_time': training_time,
            'threshold': self.threshold
        }
    
    def predict_anomaly(
        self,
        data: np.ndarray,
        return_reconstruction: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict anomalies in vibration data."""
        
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        if self.threshold is None:
            raise ValueError("Threshold not set. Call train() first.")
        
        # Get reconstructions
        reconstructions = self.model.predict(data, verbose=0)
        
        # Calculate reconstruction errors
        mse = np.mean(np.square(data - reconstructions), axis=1)
        
        # Classify as anomaly if error exceeds threshold
        anomalies = (mse > self.threshold).astype(int)
        
        logger.debug(
            "Anomaly prediction completed",
            samples=len(data),
            anomalies_detected=np.sum(anomalies),
            avg_mse=np.mean(mse),
            threshold=self.threshold
        )
        
        if return_reconstruction:
            return anomalies, mse, reconstructions
        else:
            return anomalies, mse
    
    def export_tflite(
        self,
        output_path: str,
        quantize: bool = True
    ) -> str:
        """Export model to TensorFlow Lite format for TinyML deployment."""
        
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        logger.info(
            "Exporting model to TensorFlow Lite",
            output_path=output_path,
            quantize=quantize
        )
        
        # Convert to TensorFlow Lite
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        if quantize:
            # Enable quantization for smaller model size
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
        
        tflite_model = converter.convert()
        
        # Save to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        # Save metadata
        metadata = {
            'input_dim': self.input_dim,
            'encoding_dim': self.encoding_dim,
            'threshold': float(self.threshold) if self.threshold is not None else None,
            'model_size_bytes': len(tflite_model),
            'quantized': quantize
        }
        
        metadata_path = output_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(
            "TensorFlow Lite export completed",
            model_size=f"{len(tflite_model) / 1024:.1f} KB",
            output_path=str(output_path),
            metadata_path=str(metadata_path)
        )
        
        return str(output_path)
    
    def load_tflite(self, model_path: str) -> None:
        """Load TensorFlow Lite model for inference."""
        
        logger.info("Loading TensorFlow Lite model", path=model_path)
        
        # Load the TFLite model
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Load metadata if available
        metadata_path = Path(model_path).with_suffix('.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.threshold = metadata.get('threshold')
                self.input_dim = metadata.get('input_dim', self.input_dim)
                self.encoding_dim = metadata.get('encoding_dim', self.encoding_dim)
        
        logger.info("TensorFlow Lite model loaded successfully")
    
    def tflite_predict(self, data: np.ndarray) -> np.ndarray:
        """Run inference using TensorFlow Lite model."""
        
        if not hasattr(self, 'interpreter'):
            raise ValueError("TFLite model not loaded. Call load_tflite() first.")
        
        # Prepare input data
        input_data = data.astype(np.float32)
        if len(input_data.shape) == 1:
            input_data = input_data.reshape(1, -1)
        
        # Run inference
        predictions = []
        for sample in input_data:
            self.interpreter.set_tensor(
                self.input_details[0]['index'], 
                sample.reshape(1, -1)
            )
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(self.output_details[0]['index'])
            predictions.append(output[0])
        
        return np.array(predictions)
    
    def benchmark_inference(
        self,
        test_data: np.ndarray,
        num_runs: int = 100
    ) -> Dict:
        """Benchmark inference performance."""
        
        logger.info(
            "Starting inference benchmark",
            test_samples=len(test_data),
            runs=num_runs
        )
        
        # Benchmark full model
        if self.model is not None:
            start_time = time.time()
            for _ in range(num_runs):
                _ = self.model.predict(test_data[:1], verbose=0)
            full_model_time = (time.time() - start_time) / num_runs
        else:
            full_model_time = None
        
        # Benchmark TFLite model
        if hasattr(self, 'interpreter'):
            start_time = time.time()
            for _ in range(num_runs):
                _ = self.tflite_predict(test_data[:1])
            tflite_time = (time.time() - start_time) / num_runs
        else:
            tflite_time = None
        
        results = {
            'full_model_inference_ms': full_model_time * 1000 if full_model_time else None,
            'tflite_inference_ms': tflite_time * 1000 if tflite_time else None,
            'speedup': full_model_time / tflite_time if (full_model_time and tflite_time) else None,
            'test_samples': len(test_data),
            'runs': num_runs
        }
        
        logger.info(
            "Inference benchmark completed",
            **{k: f"{v:.2f}" if isinstance(v, float) else v for k, v in results.items()}
        )
        
        return results


class AutoencoderManager:
    """Manager class for autoencoder training and deployment."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize autoencoder manager."""
        self.config = config or {}
        self.autoencoder = None
        
        logger.info("Autoencoder manager initialized", config=self.config)
    
    def create_autoencoder(self, **kwargs) -> VibrationAutoencoder:
        """Create a new autoencoder instance."""
        
        # Merge config with kwargs
        params = {**self.config, **kwargs}
        
        self.autoencoder = VibrationAutoencoder(**params)
        return self.autoencoder
    
    def train_and_export(
        self,
        output_dir: str = "models/autoencoder",
        **kwargs
    ) -> Dict:
        """Complete training and export pipeline."""
        
        if self.autoencoder is None:
            self.create_autoencoder()
        
        # Generate training data
        train_data, labels = self.autoencoder.generate_synthetic_vibration_data(
            num_samples=kwargs.get('num_samples', 2000),
            anomaly_ratio=0.05  # Only normal data for training
        )
        
        # Use only normal samples for training
        normal_indices = labels == 0
        train_normal = train_data[normal_indices]
        
        # Split for validation
        split_idx = int(0.8 * len(train_normal))
        train_split = train_normal[:split_idx]
        val_split = train_normal[split_idx:]
        
        # Train the model
        training_results = self.autoencoder.train(
            train_split,
            validation_data=val_split,
            epochs=kwargs.get('epochs', 50),
            batch_size=kwargs.get('batch_size', 32)
        )
        
        # Export to TFLite
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        tflite_path = self.autoencoder.export_tflite(
            output_path / "vibration_autoencoder.tflite",
            quantize=True
        )
        
        # Test on mixed data
        test_data, test_labels = self.autoencoder.generate_synthetic_vibration_data(
            num_samples=500,
            anomaly_ratio=0.2,
            seed=42
        )
        
        anomalies, mse = self.autoencoder.predict_anomaly(test_data)
        
        # Calculate performance metrics
        from sklearn.metrics import classification_report, confusion_matrix
        
        results = {
            'training_results': training_results,
            'tflite_path': tflite_path,
            'test_accuracy': np.mean(anomalies == test_labels),
            'classification_report': classification_report(test_labels, anomalies, output_dict=True),
            'confusion_matrix': confusion_matrix(test_labels, anomalies).tolist()
        }
        
        logger.info(
            "Training and export completed",
            accuracy=results['test_accuracy'],
            tflite_path=tflite_path
        )
        
        return results


if __name__ == "__main__":
    # Example usage
    manager = AutoencoderManager()
    results = manager.train_and_export()
    print(f"Training completed with accuracy: {results['test_accuracy']:.3f}")

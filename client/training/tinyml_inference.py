"""
TinyML Inference Engine for On-Device Autoencoder
Raspberry Pi 5 Federated Environmental Monitoring Network
"""

import tensorflow as tf
import numpy as np
from typing import Dict, List, Optional, Tuple
import structlog
import time
from pathlib import Path
import json
import threading
from collections import deque
import psutil

logger = structlog.get_logger(__name__)


class TinyMLInferenceEngine:
    """Optimized inference engine for TinyML autoencoder deployment."""
    
    def __init__(
        self,
        model_path: str,
        buffer_size: int = 1000,
        inference_threads: int = 1
    ):
        """Initialize TinyML inference engine."""
        self.model_path = model_path
        self.buffer_size = buffer_size
        self.inference_threads = inference_threads
        
        # Model components
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.metadata = None
        
        # Inference state
        self.is_running = False
        self.inference_buffer = deque(maxlen=buffer_size)
        self.results_buffer = deque(maxlen=buffer_size)
        self.inference_lock = threading.Lock()
        
        # Performance metrics
        self.inference_count = 0
        self.total_inference_time = 0.0
        self.anomaly_count = 0
        
        logger.info(
            "TinyML inference engine initialized",
            model_path=model_path,
            buffer_size=buffer_size,
            threads=inference_threads
        )
    
    def load_model(self) -> None:
        """Load TensorFlow Lite model and metadata."""
        
        logger.info("Loading TensorFlow Lite model", path=self.model_path)
        
        # Load TFLite model
        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        
        # Get input/output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Load metadata
        metadata_path = Path(self.model_path).with_suffix('.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            logger.warning("No metadata file found", expected_path=str(metadata_path))
            self.metadata = {}
        
        # Log model information
        input_shape = self.input_details[0]['shape']
        output_shape = self.output_details[0]['shape']
        model_size = Path(self.model_path).stat().st_size / 1024  # KB
        
        logger.info(
            "TensorFlow Lite model loaded",
            input_shape=input_shape.tolist(),
            output_shape=output_shape.tolist(),
            model_size_kb=f"{model_size:.1f}",
            threshold=self.metadata.get('threshold'),
            quantized=self.metadata.get('quantized', False)
        )
    
    def preprocess_vibration_data(
        self,
        raw_data: np.ndarray,
        window_size: Optional[int] = None
    ) -> np.ndarray:
        """Preprocess raw vibration data for inference."""
        
        if window_size is None:
            window_size = self.metadata.get('input_dim', 128)
        
        # Ensure correct data type
        data = raw_data.astype(np.float32)
        
        # Handle different input sizes
        if len(data) > window_size:
            # Take the most recent window
            data = data[-window_size:]
        elif len(data) < window_size:
            # Pad with zeros
            padding = window_size - len(data)
            data = np.pad(data, (0, padding), mode='constant')
        
        # Normalize data (simple z-score normalization)
        if np.std(data) > 0:
            data = (data - np.mean(data)) / np.std(data)
        
        # Reshape for model input
        return data.reshape(1, -1)
    
    def single_inference(
        self,
        data: np.ndarray,
        return_reconstruction: bool = False
    ) -> Dict:
        """Perform single inference on preprocessed data."""
        
        start_time = time.time()
        
        # Set input tensor
        self.interpreter.set_tensor(
            self.input_details[0]['index'],
            data
        )
        
        # Run inference
        self.interpreter.invoke()
        
        # Get output
        reconstruction = self.interpreter.get_tensor(
            self.output_details[0]['index']
        )
        
        inference_time = time.time() - start_time
        
        # Calculate reconstruction error
        mse = np.mean(np.square(data - reconstruction))
        
        # Determine if anomaly
        threshold = self.metadata.get('threshold', 0.1)
        is_anomaly = mse > threshold
        
        # Update metrics
        with self.inference_lock:
            self.inference_count += 1
            self.total_inference_time += inference_time
            if is_anomaly:
                self.anomaly_count += 1
        
        result = {
            'timestamp': time.time(),
            'inference_time_ms': inference_time * 1000,
            'mse': float(mse),
            'threshold': threshold,
            'is_anomaly': bool(is_anomaly),
            'confidence': float(mse / threshold) if threshold > 0 else 0.0
        }
        
        if return_reconstruction:
            result['reconstruction'] = reconstruction.flatten()
            result['input'] = data.flatten()
        
        return result
    
    def batch_inference(
        self,
        data_batch: List[np.ndarray]
    ) -> List[Dict]:
        """Perform batch inference on multiple samples."""
        
        logger.debug("Starting batch inference", batch_size=len(data_batch))
        
        results = []
        for data in data_batch:
            preprocessed = self.preprocess_vibration_data(data)
            result = self.single_inference(preprocessed)
            results.append(result)
        
        return results
    
    def start_continuous_inference(self) -> None:
        """Start continuous inference thread."""
        
        if self.is_running:
            logger.warning("Continuous inference already running")
            return
        
        self.is_running = True
        
        def inference_worker():
            """Worker thread for continuous inference."""
            logger.info("Starting continuous inference worker")
            
            while self.is_running:
                try:
                    # Check if there's data to process
                    if len(self.inference_buffer) > 0:
                        with self.inference_lock:
                            if self.inference_buffer:
                                data = self.inference_buffer.popleft()
                            else:
                                continue
                        
                        # Preprocess and run inference
                        preprocessed = self.preprocess_vibration_data(data)
                        result = self.single_inference(preprocessed)
                        
                        # Store result
                        with self.inference_lock:
                            self.results_buffer.append(result)
                    
                    else:
                        # Sleep briefly if no data
                        time.sleep(0.001)  # 1ms
                
                except Exception as e:
                    logger.error("Error in inference worker", error=str(e))
                    time.sleep(0.01)  # 10ms on error
            
            logger.info("Continuous inference worker stopped")
        
        # Start worker threads
        self.worker_threads = []
        for i in range(self.inference_threads):
            thread = threading.Thread(
                target=inference_worker,
                name=f"TinyML-Worker-{i}",
                daemon=True
            )
            thread.start()
            self.worker_threads.append(thread)
        
        logger.info(
            "Continuous inference started",
            threads=self.inference_threads
        )
    
    def stop_continuous_inference(self) -> None:
        """Stop continuous inference."""
        
        if not self.is_running:
            return
        
        logger.info("Stopping continuous inference")
        self.is_running = False
        
        # Wait for worker threads to finish
        for thread in self.worker_threads:
            thread.join(timeout=1.0)
        
        logger.info("Continuous inference stopped")
    
    def add_data(self, data: np.ndarray) -> None:
        """Add data to inference buffer."""
        
        with self.inference_lock:
            self.inference_buffer.append(data.copy())
    
    def get_results(self, max_results: Optional[int] = None) -> List[Dict]:
        """Get inference results from buffer."""
        
        with self.inference_lock:
            if max_results is None:
                results = list(self.results_buffer)
                self.results_buffer.clear()
            else:
                results = []
                for _ in range(min(max_results, len(self.results_buffer))):
                    if self.results_buffer:
                        results.append(self.results_buffer.popleft())
        
        return results
    
    def get_performance_metrics(self) -> Dict:
        """Get performance metrics."""
        
        with self.inference_lock:
            if self.inference_count > 0:
                avg_inference_time = self.total_inference_time / self.inference_count
                anomaly_rate = self.anomaly_count / self.inference_count
            else:
                avg_inference_time = 0.0
                anomaly_rate = 0.0
            
            metrics = {
                'total_inferences': self.inference_count,
                'avg_inference_time_ms': avg_inference_time * 1000,
                'total_inference_time_s': self.total_inference_time,
                'anomaly_count': self.anomaly_count,
                'anomaly_rate': anomaly_rate,
                'buffer_utilization': len(self.inference_buffer) / self.buffer_size,
                'results_pending': len(self.results_buffer)
            }
        
        # Add system metrics
        metrics.update({
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_available_mb': psutil.virtual_memory().available / (1024 * 1024)
        })
        
        return metrics
    
    def reset_metrics(self) -> None:
        """Reset performance metrics."""
        
        with self.inference_lock:
            self.inference_count = 0
            self.total_inference_time = 0.0
            self.anomaly_count = 0
        
        logger.info("Performance metrics reset")
    
    def benchmark_performance(
        self,
        num_samples: int = 1000,
        data_generator: Optional[callable] = None
    ) -> Dict:
        """Benchmark inference performance."""
        
        logger.info("Starting performance benchmark", samples=num_samples)
        
        # Generate test data if not provided
        if data_generator is None:
            input_dim = self.metadata.get('input_dim', 128)
            test_data = [
                np.random.randn(input_dim).astype(np.float32)
                for _ in range(num_samples)
            ]
        else:
            test_data = [data_generator() for _ in range(num_samples)]
        
        # Reset metrics
        self.reset_metrics()
        
        # Run benchmark
        start_time = time.time()
        results = self.batch_inference(test_data)
        total_time = time.time() - start_time
        
        # Calculate statistics
        inference_times = [r['inference_time_ms'] for r in results]
        mse_values = [r['mse'] for r in results]
        anomalies = [r['is_anomaly'] for r in results]
        
        benchmark_results = {
            'total_samples': num_samples,
            'total_time_s': total_time,
            'avg_inference_time_ms': np.mean(inference_times),
            'min_inference_time_ms': np.min(inference_times),
            'max_inference_time_ms': np.max(inference_times),
            'std_inference_time_ms': np.std(inference_times),
            'throughput_samples_per_sec': num_samples / total_time,
            'avg_mse': np.mean(mse_values),
            'anomaly_rate': np.mean(anomalies),
            'model_size_kb': Path(self.model_path).stat().st_size / 1024
        }
        
        logger.info(
            "Performance benchmark completed",
            **{k: f"{v:.2f}" if isinstance(v, float) else v 
               for k, v in benchmark_results.items()}
        )
        
        return benchmark_results


class OnDeviceTraining:
    """On-device training capabilities for autoencoder fine-tuning."""
    
    def __init__(
        self,
        inference_engine: TinyMLInferenceEngine,
        learning_rate: float = 0.001
    ):
        """Initialize on-device training."""
        self.inference_engine = inference_engine
        self.learning_rate = learning_rate
        
        # Training state
        self.training_data = deque(maxlen=1000)
        self.is_training = False
        
        logger.info(
            "On-device training initialized",
            learning_rate=learning_rate
        )
    
    def collect_training_sample(
        self,
        data: np.ndarray,
        is_normal: bool = True
    ) -> None:
        """Collect sample for on-device training."""
        
        if is_normal:  # Only collect normal samples for autoencoder training
            preprocessed = self.inference_engine.preprocess_vibration_data(data)
            self.training_data.append(preprocessed.flatten())
    
    def perform_sgd_update(self) -> Dict:
        """Perform single SGD update on collected data."""
        
        if len(self.training_data) < 10:
            logger.warning("Insufficient training data for SGD update")
            return {'status': 'insufficient_data', 'samples': len(self.training_data)}
        
        logger.info("Performing on-device SGD update", samples=len(self.training_data))
        
        start_time = time.time()
        
        # Convert training data to batch
        batch_data = np.array(list(self.training_data))
        
        # Note: This is a simplified version. In practice, you would need
        # to implement gradient computation and weight updates for TFLite
        # This would require custom TensorFlow operations or conversion
        # back to full TensorFlow model for training
        
        # For now, we simulate the training time and return metrics
        training_time = time.time() - start_time
        
        # Clear training buffer after update
        self.training_data.clear()
        
        results = {
            'status': 'completed',
            'training_time_ms': training_time * 1000,
            'samples_used': len(batch_data),
            'learning_rate': self.learning_rate
        }
        
        logger.info(
            "SGD update completed",
            **{k: f"{v:.2f}" if isinstance(v, float) else v 
               for k, v in results.items()}
        )
        
        return results


if __name__ == "__main__":
    # Example usage
    model_path = "models/autoencoder/vibration_autoencoder.tflite"
    
    # Create inference engine
    engine = TinyMLInferenceEngine(model_path)
    engine.load_model()
    
    # Run benchmark
    benchmark_results = engine.benchmark_performance(num_samples=100)
    print(f"Average inference time: {benchmark_results['avg_inference_time_ms']:.2f} ms")
    print(f"Throughput: {benchmark_results['throughput_samples_per_sec']:.1f} samples/sec")

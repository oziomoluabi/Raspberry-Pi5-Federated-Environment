#!/usr/bin/env python3
"""
Performance Benchmarking Script
Raspberry Pi 5 Federated Environmental Monitoring Network

Comprehensive performance benchmarking for Sprint 6 CI/CD pipeline:
- Federated learning performance metrics
- TinyML inference benchmarks
- MATLAB integration performance
- Memory and CPU profiling
- Network communication benchmarks
"""

import sys
import time
import json
import argparse
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np
import pandas as pd
import psutil
import structlog

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from client.training.autoencoder import VibrationAutoencoder
from client.training.tinyml_inference import TinyMLInferenceEngine
from server.models.lstm_model import create_lstm_model, LSTMModelManager

# Configure structured logging
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
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


class PerformanceMonitor:
    """Monitor system performance during benchmarks."""
    
    def __init__(self):
        self.monitoring = False
        self.metrics = {
            'cpu_percent': [],
            'memory_mb': [],
            'disk_io': [],
            'network_io': [],
            'timestamps': []
        }
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start performance monitoring."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        logger.info("Performance monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=None)
                
                # Memory usage
                memory = psutil.virtual_memory()
                memory_mb = memory.used / 1024 / 1024
                
                # Disk I/O
                disk_io = psutil.disk_io_counters()
                
                # Network I/O
                network_io = psutil.net_io_counters()
                
                # Store metrics
                self.metrics['cpu_percent'].append(cpu_percent)
                self.metrics['memory_mb'].append(memory_mb)
                self.metrics['disk_io'].append({
                    'read_bytes': disk_io.read_bytes if disk_io else 0,
                    'write_bytes': disk_io.write_bytes if disk_io else 0
                })
                self.metrics['network_io'].append({
                    'bytes_sent': network_io.bytes_sent if network_io else 0,
                    'bytes_recv': network_io.bytes_recv if network_io else 0
                })
                self.metrics['timestamps'].append(time.time())
                
                time.sleep(0.1)  # Monitor every 100ms
                
            except Exception as e:
                logger.error("Error in monitoring loop", error=str(e))
                break
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        if not self.metrics['cpu_percent']:
            return {}
        
        return {
            'cpu': {
                'mean': np.mean(self.metrics['cpu_percent']),
                'max': np.max(self.metrics['cpu_percent']),
                'min': np.min(self.metrics['cpu_percent']),
                'std': np.std(self.metrics['cpu_percent'])
            },
            'memory': {
                'mean_mb': np.mean(self.metrics['memory_mb']),
                'max_mb': np.max(self.metrics['memory_mb']),
                'min_mb': np.min(self.metrics['memory_mb']),
                'std_mb': np.std(self.metrics['memory_mb'])
            },
            'duration_seconds': self.metrics['timestamps'][-1] - self.metrics['timestamps'][0] if len(self.metrics['timestamps']) > 1 else 0,
            'samples': len(self.metrics['timestamps'])
        }


class FederatedLearningBenchmark:
    """Benchmark federated learning performance."""
    
    def __init__(self):
        self.results = {}
    
    def benchmark_lstm_training(self, num_clients: int = 3, rounds: int = 3) -> Dict[str, Any]:
        """Benchmark LSTM model training performance."""
        logger.info("Starting LSTM training benchmark", num_clients=num_clients, rounds=rounds)
        
        # Generate synthetic data
        sequence_length = 10
        n_features = 3
        data_size = 1000
        
        X = np.random.randn(data_size, sequence_length, n_features)
        y = np.random.randn(data_size, n_features)
        
        # Split data among clients
        client_data = []
        samples_per_client = data_size // num_clients
        for i in range(num_clients):
            start_idx = i * samples_per_client
            end_idx = (i + 1) * samples_per_client
            client_data.append((X[start_idx:end_idx], y[start_idx:end_idx]))
        
        # Benchmark training
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        start_time = time.time()
        
        # Initialize global model
        global_model = create_lstm_model(
            sequence_length=sequence_length,
            n_features=n_features,
            lstm_units=64
        )
        
        round_times = []
        
        for round_num in range(rounds):
            round_start = time.time()
            
            # Train on each client
            client_weights = []
            for client_idx, (X_client, y_client) in enumerate(client_data):
                # Create local model with global weights
                local_model = create_lstm_model(
                    sequence_length=sequence_length,
                    n_features=n_features,
                    lstm_units=64
                )
                local_model.set_weights(global_model.get_weights())
                
                # Local training
                local_model.fit(
                    X_client, y_client,
                    epochs=2,
                    batch_size=32,
                    verbose=0
                )
                
                client_weights.append(local_model.get_weights())
            
            # Federated averaging
            averaged_weights = []
            for layer_idx in range(len(client_weights[0])):
                layer_weights = [weights[layer_idx] for weights in client_weights]
                avg_weights = np.mean(layer_weights, axis=0)
                averaged_weights.append(avg_weights)
            
            global_model.set_weights(averaged_weights)
            
            round_time = time.time() - round_start
            round_times.append(round_time)
            
            logger.info("Completed federated round", round=round_num+1, time_seconds=round_time)
        
        total_time = time.time() - start_time
        monitor.stop_monitoring()
        
        # Evaluate final model
        test_loss = global_model.evaluate(X[:100], y[:100], verbose=0)
        
        results = {
            'total_time_seconds': total_time,
            'average_round_time_seconds': np.mean(round_times),
            'rounds': rounds,
            'num_clients': num_clients,
            'final_loss': float(test_loss),
            'performance_metrics': monitor.get_summary(),
            'round_times': round_times
        }
        
        self.results['lstm_training'] = results
        return results
    
    def benchmark_model_aggregation(self, num_models: int = 10, model_size: int = 1000000) -> Dict[str, Any]:
        """Benchmark model parameter aggregation performance."""
        logger.info("Starting model aggregation benchmark", num_models=num_models, model_size=model_size)
        
        # Generate random model parameters
        model_params = []
        for _ in range(num_models):
            params = [np.random.randn(model_size // 4).astype(np.float32) for _ in range(4)]
            model_params.append(params)
        
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        start_time = time.time()
        
        # Perform aggregation
        aggregated_params = []
        for layer_idx in range(len(model_params[0])):
            layer_weights = [params[layer_idx] for params in model_params]
            avg_weights = np.mean(layer_weights, axis=0)
            aggregated_params.append(avg_weights)
        
        aggregation_time = time.time() - start_time
        monitor.stop_monitoring()
        
        results = {
            'aggregation_time_seconds': aggregation_time,
            'num_models': num_models,
            'model_size_parameters': model_size,
            'throughput_params_per_second': model_size * num_models / aggregation_time,
            'performance_metrics': monitor.get_summary()
        }
        
        self.results['model_aggregation'] = results
        return results


class TinyMLBenchmark:
    """Benchmark TinyML inference performance."""
    
    def __init__(self):
        self.results = {}
    
    def benchmark_autoencoder_training(self, data_size: int = 1000, epochs: int = 10) -> Dict[str, Any]:
        """Benchmark autoencoder training performance."""
        logger.info("Starting autoencoder training benchmark", data_size=data_size, epochs=epochs)
        
        # Generate synthetic vibration data
        vibration_data = np.random.normal(0, 0.1, (data_size, 3)).astype(np.float32)
        
        # Add some anomalies
        anomaly_indices = np.random.choice(data_size, data_size // 10, replace=False)
        vibration_data[anomaly_indices] += np.random.normal(0, 1.0, (len(anomaly_indices), 3))
        
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        start_time = time.time()
        
        # Create and train autoencoder
        autoencoder = VibrationAutoencoder(
            input_dim=3,
            encoding_dim=2,
            learning_rate=0.001
        )
        
        model = autoencoder.build_model()
        
        history = autoencoder.train(
            vibration_data,
            epochs=epochs,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        
        training_time = time.time() - start_time
        monitor.stop_monitoring()
        
        # Test inference performance
        inference_times = []
        for _ in range(100):
            test_sample = vibration_data[np.random.randint(0, len(vibration_data))].reshape(1, -1)
            
            inf_start = time.time()
            prediction = model.predict(test_sample, verbose=0)
            inf_time = time.time() - inf_start
            
            inference_times.append(inf_time)
        
        results = {
            'training_time_seconds': training_time,
            'data_size': data_size,
            'epochs': epochs,
            'final_loss': float(history.history['loss'][-1]),
            'average_inference_time_ms': np.mean(inference_times) * 1000,
            'inference_throughput_samples_per_second': 1.0 / np.mean(inference_times),
            'performance_metrics': monitor.get_summary()
        }
        
        self.results['autoencoder_training'] = results
        return results
    
    def benchmark_tflite_inference(self, num_inferences: int = 1000) -> Dict[str, Any]:
        """Benchmark TensorFlow Lite inference performance."""
        logger.info("Starting TFLite inference benchmark", num_inferences=num_inferences)
        
        # Create and train a simple autoencoder
        autoencoder = VibrationAutoencoder(input_dim=3, encoding_dim=2)
        model = autoencoder.build_model()
        
        # Train briefly
        dummy_data = np.random.normal(0, 0.1, (100, 3)).astype(np.float32)
        model.fit(dummy_data, dummy_data, epochs=2, verbose=0)
        
        # Convert to TensorFlow Lite
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.tflite', delete=False) as tmp_file:
            tflite_path = tmp_file.name
            autoencoder.convert_to_tflite(tflite_path)
        
        # Initialize inference engine
        inference_engine = TinyMLInferenceEngine(tflite_path)
        
        # Benchmark inference
        test_data = np.random.normal(0, 0.1, (num_inferences, 3)).astype(np.float32)
        
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        inference_times = []
        start_time = time.time()
        
        for i in range(num_inferences):
            sample = test_data[i:i+1]
            
            inf_start = time.time()
            prediction = inference_engine.predict(sample)
            inf_time = time.time() - inf_start
            
            inference_times.append(inf_time)
        
        total_time = time.time() - start_time
        monitor.stop_monitoring()
        
        # Cleanup
        Path(tflite_path).unlink()
        
        results = {
            'total_time_seconds': total_time,
            'num_inferences': num_inferences,
            'average_inference_time_ms': np.mean(inference_times) * 1000,
            'min_inference_time_ms': np.min(inference_times) * 1000,
            'max_inference_time_ms': np.max(inference_times) * 1000,
            'throughput_inferences_per_second': num_inferences / total_time,
            'performance_metrics': monitor.get_summary()
        }
        
        self.results['tflite_inference'] = results
        return results


class MATLABBenchmark:
    """Benchmark MATLAB integration performance."""
    
    def __init__(self):
        self.results = {}
    
    def benchmark_data_processing(self, data_size: int = 10000) -> Dict[str, Any]:
        """Benchmark MATLAB data processing performance."""
        logger.info("Starting MATLAB data processing benchmark", data_size=data_size)
        
        # Generate synthetic environmental data
        timestamps = pd.date_range(start='2024-01-01', periods=data_size, freq='1min')
        environmental_data = pd.DataFrame({
            'timestamp': timestamps,
            'temperature': np.random.normal(22.0, 2.0, data_size),
            'humidity': np.random.normal(45.0, 10.0, data_size),
            'pressure': np.random.normal(1013.25, 5.0, data_size)
        })
        
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        start_time = time.time()
        
        # Simulate MATLAB processing (using NumPy as fallback)
        try:
            from client.matlab.matlab_integration import EnvironmentalDataProcessor
            processor = EnvironmentalDataProcessor()
            
            # Mock MATLAB engine initialization
            processor.engine_manager.engine = type('MockEngine', (), {
                'eval': lambda self, x: None,
                'mean': lambda self, x: np.mean(x),
                'std': lambda self, x: np.std(x),
                'env_preprocess': lambda self, data: {
                    'processed_temp': np.array(data['temperature']),
                    'processed_humidity': np.array(data['humidity']),
                    'features': np.column_stack([
                        data['temperature'], data['humidity'], data['pressure']
                    ])
                }
            })()
            processor.engine_manager.engine_type = 'matlab'
            
            # Process data
            processed_data = processor.preprocess_data(environmental_data)
            
        except ImportError:
            # Fallback to NumPy processing
            processed_data = {
                'processed_temp': environmental_data['temperature'].values,
                'processed_humidity': environmental_data['humidity'].values,
                'features': environmental_data[['temperature', 'humidity', 'pressure']].values
            }
        
        processing_time = time.time() - start_time
        monitor.stop_monitoring()
        
        results = {
            'processing_time_seconds': processing_time,
            'data_size': data_size,
            'throughput_samples_per_second': data_size / processing_time,
            'features_shape': processed_data['features'].shape if 'features' in processed_data else None,
            'performance_metrics': monitor.get_summary()
        }
        
        self.results['matlab_processing'] = results
        return results


class ComprehensiveBenchmark:
    """Run comprehensive performance benchmarks."""
    
    def __init__(self, output_file: Optional[str] = None):
        self.output_file = output_file
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'system_info': self._get_system_info(),
            'benchmarks': {}
        }
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        return {
            'cpu_count': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'python_version': sys.version,
            'platform': sys.platform
        }
    
    def run_all_benchmarks(self):
        """Run all performance benchmarks."""
        logger.info("Starting comprehensive performance benchmarks")
        
        # Federated Learning Benchmarks
        fl_benchmark = FederatedLearningBenchmark()
        self.results['benchmarks']['federated_learning'] = {
            'lstm_training': fl_benchmark.benchmark_lstm_training(),
            'model_aggregation': fl_benchmark.benchmark_model_aggregation()
        }
        
        # TinyML Benchmarks
        tinyml_benchmark = TinyMLBenchmark()
        self.results['benchmarks']['tinyml'] = {
            'autoencoder_training': tinyml_benchmark.benchmark_autoencoder_training(),
            'tflite_inference': tinyml_benchmark.benchmark_tflite_inference()
        }
        
        # MATLAB Benchmarks
        matlab_benchmark = MATLABBenchmark()
        self.results['benchmarks']['matlab'] = {
            'data_processing': matlab_benchmark.benchmark_data_processing()
        }
        
        logger.info("Completed all performance benchmarks")
        
        # Save results
        if self.output_file:
            self.save_results()
        
        return self.results
    
    def save_results(self):
        """Save benchmark results to file."""
        output_path = Path(self.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info("Benchmark results saved", output_file=self.output_file)
    
    def print_summary(self):
        """Print benchmark summary."""
        print("\n" + "="*80)
        print("PERFORMANCE BENCHMARK SUMMARY")
        print("="*80)
        
        # System Info
        print(f"\nSystem Information:")
        print(f"  CPU Cores: {self.results['system_info']['cpu_count']}")
        print(f"  Memory: {self.results['system_info']['memory_total_gb']:.1f} GB")
        print(f"  Platform: {self.results['system_info']['platform']}")
        
        # Federated Learning Results
        fl_results = self.results['benchmarks']['federated_learning']
        print(f"\nFederated Learning Performance:")
        print(f"  LSTM Training Time: {fl_results['lstm_training']['total_time_seconds']:.2f}s")
        print(f"  Average Round Time: {fl_results['lstm_training']['average_round_time_seconds']:.2f}s")
        print(f"  Model Aggregation: {fl_results['model_aggregation']['aggregation_time_seconds']:.2f}s")
        
        # TinyML Results
        tinyml_results = self.results['benchmarks']['tinyml']
        print(f"\nTinyML Performance:")
        print(f"  Autoencoder Training: {tinyml_results['autoencoder_training']['training_time_seconds']:.2f}s")
        print(f"  TFLite Inference: {tinyml_results['tflite_inference']['average_inference_time_ms']:.2f}ms")
        print(f"  Inference Throughput: {tinyml_results['tflite_inference']['throughput_inferences_per_second']:.0f} samples/s")
        
        # MATLAB Results
        matlab_results = self.results['benchmarks']['matlab']
        print(f"\nMATLAB Integration Performance:")
        print(f"  Data Processing: {matlab_results['data_processing']['processing_time_seconds']:.2f}s")
        print(f"  Processing Throughput: {matlab_results['data_processing']['throughput_samples_per_second']:.0f} samples/s")
        
        print("\n" + "="*80)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Performance Benchmarking Script")
    parser.add_argument('--output', '-o', type=str, help='Output file for results')
    parser.add_argument('--benchmark', '-b', type=str, choices=['all', 'federated', 'tinyml', 'matlab'], 
                       default='all', help='Benchmark to run')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Set up logging level
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run benchmarks
    benchmark = ComprehensiveBenchmark(output_file=args.output)
    
    if args.benchmark == 'all':
        results = benchmark.run_all_benchmarks()
    elif args.benchmark == 'federated':
        fl_benchmark = FederatedLearningBenchmark()
        results = {
            'federated_learning': {
                'lstm_training': fl_benchmark.benchmark_lstm_training(),
                'model_aggregation': fl_benchmark.benchmark_model_aggregation()
            }
        }
    elif args.benchmark == 'tinyml':
        tinyml_benchmark = TinyMLBenchmark()
        results = {
            'tinyml': {
                'autoencoder_training': tinyml_benchmark.benchmark_autoencoder_training(),
                'tflite_inference': tinyml_benchmark.benchmark_tflite_inference()
            }
        }
    elif args.benchmark == 'matlab':
        matlab_benchmark = MATLABBenchmark()
        results = {
            'matlab': {
                'data_processing': matlab_benchmark.benchmark_data_processing()
            }
        }
    
    # Print summary
    benchmark.print_summary()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

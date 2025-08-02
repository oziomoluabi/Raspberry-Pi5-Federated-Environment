#!/usr/bin/env python3
"""
TinyML Autoencoder Pipeline Test Script
Raspberry Pi 5 Federated Environmental Monitoring Network

This script demonstrates the complete TinyML pipeline:
1. Train autoencoder on synthetic vibration data
2. Export to TensorFlow Lite
3. Run inference benchmarks
4. Simulate on-device deployment
"""

import sys
import time
import argparse
from pathlib import Path
import structlog

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from client.training.autoencoder import VibrationAutoencoder, AutoencoderManager
from client.training.tinyml_inference import TinyMLInferenceEngine, OnDeviceTraining
from client.sensing.vibration_sensor import ADXL345VibrationSensor, VibrationMonitor

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


def train_autoencoder_model(
    output_dir: str = "models/autoencoder",
    input_dim: int = 128,
    encoding_dim: int = 32,
    num_samples: int = 2000,
    epochs: int = 50
) -> str:
    """Train and export autoencoder model."""
    
    logger.info(
        "Starting autoencoder training",
        output_dir=output_dir,
        input_dim=input_dim,
        encoding_dim=encoding_dim,
        samples=num_samples,
        epochs=epochs
    )
    
    # Create autoencoder manager
    manager = AutoencoderManager({
        'input_dim': input_dim,
        'encoding_dim': encoding_dim,
        'learning_rate': 0.001
    })
    
    # Train and export
    results = manager.train_and_export(
        output_dir=output_dir,
        num_samples=num_samples,
        epochs=epochs,
        batch_size=32
    )
    
    logger.info(
        "Autoencoder training completed",
        accuracy=results['test_accuracy'],
        tflite_path=results['tflite_path']
    )
    
    return results['tflite_path']


def benchmark_inference_performance(
    model_path: str,
    num_samples: int = 1000
) -> dict:
    """Benchmark TensorFlow Lite inference performance."""
    
    logger.info(
        "Starting inference benchmark",
        model_path=model_path,
        samples=num_samples
    )
    
    # Create inference engine
    engine = TinyMLInferenceEngine(model_path)
    engine.load_model()
    
    # Run benchmark
    results = engine.benchmark_performance(num_samples=num_samples)
    
    logger.info(
        "Inference benchmark completed",
        avg_time_ms=results['avg_inference_time_ms'],
        throughput=results['throughput_samples_per_sec'],
        model_size_kb=results['model_size_kb']
    )
    
    return results


def simulate_realtime_processing(
    model_path: str,
    duration: int = 30,
    sample_rate: int = 400
) -> dict:
    """Simulate real-time vibration processing."""
    
    logger.info(
        "Starting real-time simulation",
        model_path=model_path,
        duration=duration,
        sample_rate=sample_rate
    )
    
    # Create components
    sensor = ADXL345VibrationSensor(
        sample_rate=sample_rate,
        simulation_mode=True
    )
    
    engine = TinyMLInferenceEngine(model_path)
    engine.load_model()
    
    # Start data collection
    sensor.start_collection()
    engine.start_continuous_inference()
    
    anomaly_count = 0
    total_processed = 0
    
    def anomaly_callback(data):
        nonlocal anomaly_count
        anomaly_count += 1
        logger.info("Anomaly detected", timestamp=data['timestamp'])
    
    try:
        start_time = time.time()
        
        while time.time() - start_time < duration:
            # Get recent vibration data
            features = sensor.get_vibration_features(window_size=128)
            
            # Add to inference queue
            engine.add_data(features)
            
            # Process results
            results = engine.get_results()
            for result in results:
                total_processed += 1
                if result['is_anomaly']:
                    anomaly_callback(result)
            
            time.sleep(0.1)  # 10 Hz processing
        
        # Get final metrics
        performance_metrics = engine.get_performance_metrics()
        sensor_stats = sensor.get_statistics()
        
        simulation_results = {
            'duration': duration,
            'total_processed': total_processed,
            'anomaly_count': anomaly_count,
            'anomaly_rate': anomaly_count / total_processed if total_processed > 0 else 0,
            'performance_metrics': performance_metrics,
            'sensor_stats': sensor_stats
        }
        
        logger.info(
            "Real-time simulation completed",
            processed=total_processed,
            anomalies=anomaly_count,
            avg_inference_ms=performance_metrics['avg_inference_time_ms']
        )
        
        return simulation_results
    
    finally:
        # Cleanup
        engine.stop_continuous_inference()
        sensor.stop_collection()


def test_on_device_training(
    model_path: str,
    num_training_samples: int = 100
) -> dict:
    """Test on-device training capabilities."""
    
    logger.info(
        "Testing on-device training",
        model_path=model_path,
        training_samples=num_training_samples
    )
    
    # Create components
    engine = TinyMLInferenceEngine(model_path)
    engine.load_model()
    
    trainer = OnDeviceTraining(engine, learning_rate=0.001)
    
    # Generate training data
    sensor = ADXL345VibrationSensor(simulation_mode=True)
    sensor.start_collection()
    
    try:
        # Collect training samples
        for i in range(num_training_samples):
            features = sensor.get_vibration_features(window_size=128)
            trainer.collect_training_sample(features, is_normal=True)
            time.sleep(0.01)  # 100 Hz collection
        
        # Perform SGD update
        training_results = trainer.perform_sgd_update()
        
        logger.info(
            "On-device training completed",
            **training_results
        )
        
        return training_results
    
    finally:
        sensor.stop_collection()


def run_complete_pipeline(
    output_dir: str = "models/autoencoder",
    skip_training: bool = False
) -> dict:
    """Run the complete TinyML pipeline."""
    
    logger.info("Starting complete TinyML pipeline", output_dir=output_dir)
    
    results = {}
    
    # Step 1: Train autoencoder (if not skipping)
    model_path = Path(output_dir) / "vibration_autoencoder.tflite"
    
    if not skip_training or not model_path.exists():
        logger.info("Step 1: Training autoencoder model")
        model_path = train_autoencoder_model(
            output_dir=output_dir,
            input_dim=128,
            encoding_dim=32,
            num_samples=2000,
            epochs=30  # Reduced for faster testing
        )
        results['training'] = {'model_path': model_path}
    else:
        logger.info("Step 1: Skipping training, using existing model")
        model_path = str(model_path)
    
    # Step 2: Benchmark inference performance
    logger.info("Step 2: Benchmarking inference performance")
    benchmark_results = benchmark_inference_performance(
        model_path=model_path,
        num_samples=500
    )
    results['benchmark'] = benchmark_results
    
    # Step 3: Simulate real-time processing
    logger.info("Step 3: Simulating real-time processing")
    realtime_results = simulate_realtime_processing(
        model_path=model_path,
        duration=15,  # 15 seconds
        sample_rate=400
    )
    results['realtime'] = realtime_results
    
    # Step 4: Test on-device training
    logger.info("Step 4: Testing on-device training")
    training_results = test_on_device_training(
        model_path=model_path,
        num_training_samples=50
    )
    results['on_device_training'] = training_results
    
    # Summary
    logger.info("Complete TinyML pipeline finished")
    
    return results


def print_results_summary(results: dict):
    """Print a summary of pipeline results."""
    
    print("\n" + "="*80)
    print("TINYML AUTOENCODER PIPELINE RESULTS")
    print("="*80)
    
    if 'training' in results:
        print(f"✅ Model Training: Completed")
        print(f"   Model Path: {results['training']['model_path']}")
    
    if 'benchmark' in results:
        bench = results['benchmark']
        print(f"✅ Inference Benchmark:")
        print(f"   Average Inference Time: {bench['avg_inference_time_ms']:.2f} ms")
        print(f"   Throughput: {bench['throughput_samples_per_sec']:.1f} samples/sec")
        print(f"   Model Size: {bench['model_size_kb']:.1f} KB")
    
    if 'realtime' in results:
        rt = results['realtime']
        print(f"✅ Real-time Simulation:")
        print(f"   Samples Processed: {rt['total_processed']}")
        print(f"   Anomalies Detected: {rt['anomaly_count']}")
        print(f"   Anomaly Rate: {rt['anomaly_rate']:.1%}")
        print(f"   Avg Inference Time: {rt['performance_metrics']['avg_inference_time_ms']:.2f} ms")
    
    if 'on_device_training' in results:
        odt = results['on_device_training']
        print(f"✅ On-Device Training:")
        print(f"   Status: {odt['status']}")
        print(f"   Training Time: {odt['training_time_ms']:.2f} ms")
        print(f"   Samples Used: {odt['samples_used']}")
    
    print("="*80)
    
    # Performance assessment
    if 'benchmark' in results:
        inference_time = results['benchmark']['avg_inference_time_ms']
        if inference_time < 10:
            print("🎯 PERFORMANCE TARGET MET: Inference < 10 ms/sample")
        else:
            print(f"⚠️  PERFORMANCE TARGET MISSED: Inference {inference_time:.2f} ms > 10 ms")
    
    if 'on_device_training' in results:
        training_time = results['on_device_training']['training_time_ms']
        if training_time < 500:
            print("🎯 PERFORMANCE TARGET MET: On-device training < 500 ms")
        else:
            print(f"⚠️  PERFORMANCE TARGET MISSED: Training {training_time:.2f} ms > 500 ms")
    
    print("="*80)


def main():
    """Main function for TinyML pipeline testing."""
    
    parser = argparse.ArgumentParser(description="TinyML Autoencoder Pipeline Test")
    parser.add_argument(
        "--output-dir",
        default="models/autoencoder",
        help="Output directory for models"
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training if model exists"
    )
    parser.add_argument(
        "--benchmark-only",
        action="store_true",
        help="Only run benchmark on existing model"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        if args.benchmark_only:
            # Only run benchmark
            model_path = Path(args.output_dir) / "vibration_autoencoder.tflite"
            if not model_path.exists():
                print(f"Error: Model not found at {model_path}")
                sys.exit(1)
            
            results = {
                'benchmark': benchmark_inference_performance(str(model_path))
            }
        else:
            # Run complete pipeline
            results = run_complete_pipeline(
                output_dir=args.output_dir,
                skip_training=args.skip_training
            )
        
        # Print summary
        print_results_summary(results)
        
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
    except Exception as e:
        logger.error("Pipeline failed", error=str(e), exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

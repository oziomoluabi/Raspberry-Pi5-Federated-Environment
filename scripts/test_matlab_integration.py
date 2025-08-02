#!/usr/bin/env python3
"""
MATLAB/Simulink Integration Test Script
Raspberry Pi 5 Federated Environmental Monitoring Network

This script tests the complete MATLAB/Simulink integration pipeline:
1. MATLAB Engine initialization and fallback to Octave
2. Environmental data preprocessing with env_preprocess.m
3. Simulink model creation and execution
4. Performance benchmarking and validation
"""

import sys
import time
import argparse
import numpy as np
from pathlib import Path
import structlog

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from client.matlab.matlab_integration import (
    MATLABEngineManager,
    EnvironmentalDataProcessor,
    SimulinkModelRunner
)

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


def generate_synthetic_environmental_data(
    duration_hours: float = 24.0,
    sample_rate: float = 1.0,  # samples per hour
    add_anomalies: bool = True
) -> tuple:
    """Generate synthetic environmental data for testing."""
    
    logger.info(
        "Generating synthetic environmental data",
        duration_hours=duration_hours,
        sample_rate=sample_rate,
        add_anomalies=add_anomalies
    )
    
    # Time vector
    num_samples = int(duration_hours * sample_rate)
    timestamps = np.linspace(0, duration_hours * 3600, num_samples)
    time_hours = timestamps / 3600
    
    # Temperature pattern (seasonal + daily + noise)
    temp_base = 25  # Base temperature
    temp_seasonal = 5 * np.sin(2 * np.pi * time_hours / 24)  # Daily cycle
    temp_trend = 0.1 * time_hours  # Slight warming trend
    temp_noise = np.random.normal(0, 1, num_samples)
    temperature = temp_base + temp_seasonal + temp_trend + temp_noise
    
    # Humidity pattern (inverse correlation with temperature)
    humidity_base = 60
    humidity_seasonal = -10 * np.sin(2 * np.pi * time_hours / 24)  # Inverse daily cycle
    humidity_trend = -0.05 * time_hours  # Slight drying trend
    humidity_noise = np.random.normal(0, 2, num_samples)
    humidity = humidity_base + humidity_seasonal + humidity_trend + humidity_noise
    
    # Ensure realistic ranges
    temperature = np.clip(temperature, -10, 50)
    humidity = np.clip(humidity, 10, 95)
    
    # Add anomalies if requested
    if add_anomalies:
        # Temperature spike
        anomaly_idx = int(0.3 * num_samples)
        temperature[anomaly_idx:anomaly_idx+5] += 15
        
        # Humidity drop
        anomaly_idx = int(0.7 * num_samples)
        humidity[anomaly_idx:anomaly_idx+3] -= 20
        
        logger.info("Added temperature and humidity anomalies")
    
    logger.info(
        "Synthetic data generated",
        samples=num_samples,
        temp_range=f"{temperature.min():.1f} - {temperature.max():.1f}°C",
        humidity_range=f"{humidity.min():.1f} - {humidity.max():.1f}%"
    )
    
    return timestamps, temperature, humidity


def generate_synthetic_vibration_data(
    duration_seconds: float = 100.0,
    sample_rate: float = 100.0,  # Hz
    add_anomalies: bool = True
) -> tuple:
    """Generate synthetic vibration data for Simulink testing."""
    
    logger.info(
        "Generating synthetic vibration data",
        duration_seconds=duration_seconds,
        sample_rate=sample_rate,
        add_anomalies=add_anomalies
    )
    
    # Time vector
    num_samples = int(duration_seconds * sample_rate)
    t = np.linspace(0, duration_seconds, num_samples)
    
    # Base vibration pattern
    base_freq = 50  # Hz
    vibration = (
        0.5 * np.sin(2 * np.pi * base_freq * t) +
        0.2 * np.sin(2 * np.pi * 2 * base_freq * t) +
        0.1 * np.random.normal(0, 1, num_samples)
    )
    
    # Add degradation over time
    degradation_factor = 1 + 0.5 * (t / duration_seconds) ** 2
    vibration *= degradation_factor
    
    # Add anomalies if requested
    if add_anomalies:
        anomaly_times = [30, 60, 85]
        for anomaly_time in anomaly_times:
            anomaly_idx = np.abs(t - anomaly_time) < 2
            vibration[anomaly_idx] += 2 * np.random.normal(0, 1, np.sum(anomaly_idx))
        
        logger.info("Added vibration anomalies", anomaly_times=anomaly_times)
    
    logger.info(
        "Vibration data generated",
        samples=num_samples,
        amplitude_range=f"{vibration.min():.2f} - {vibration.max():.2f}"
    )
    
    return t, vibration


def test_matlab_engine_initialization(
    matlab_path: str,
    prefer_matlab: bool = True
) -> MATLABEngineManager:
    """Test MATLAB Engine initialization with fallback."""
    
    logger.info(
        "Testing MATLAB Engine initialization",
        matlab_path=matlab_path,
        prefer_matlab=prefer_matlab
    )
    
    # Create manager
    manager = MATLABEngineManager(
        matlab_path=matlab_path,
        prefer_matlab=prefer_matlab,
        startup_timeout=30.0
    )
    
    # Initialize engines
    start_time = time.time()
    success = manager.initialize_engines()
    init_time = time.time() - start_time
    
    if success:
        logger.info(
            "Engine initialization successful",
            engine_type=manager.engine_type,
            init_time=f"{init_time:.2f}s"
        )
        
        # Test basic functionality
        result = manager.evaluate_expression("2 + 2")
        if result.success and result.data == 4:
            logger.info("Basic functionality test passed")
        else:
            logger.error("Basic functionality test failed")
            
        # Test array operations
        test_array = np.array([1, 2, 3, 4, 5])
        manager.set_variable("test_array", test_array)
        result = manager.evaluate_expression("sum(test_array)")
        
        if result.success and result.data == 15:
            logger.info("Array operations test passed")
        else:
            logger.error("Array operations test failed")
            
        return manager
    else:
        logger.error("Engine initialization failed")
        return None


def test_environmental_data_processing(
    manager: MATLABEngineManager,
    duration_hours: float = 24.0
) -> dict:
    """Test environmental data preprocessing."""
    
    logger.info("Testing environmental data processing")
    
    # Generate test data
    timestamps, temperature, humidity = generate_synthetic_environmental_data(
        duration_hours=duration_hours,
        sample_rate=1.0,  # 1 sample per hour
        add_anomalies=True
    )
    
    # Create processor
    processor = EnvironmentalDataProcessor(manager)
    
    # Set processing options
    options = {
        'window_size': 6,  # 6-hour moving average
        'outlier_threshold': 2.5,
        'forecast_horizon': 12,  # 12-hour forecast
        'sampling_rate': 1.0,
        'enable_plots': False  # Disable plots for automated testing
    }
    
    # Process data
    start_time = time.time()
    result = processor.process_environmental_data(
        temperature, humidity, timestamps, options
    )
    processing_time = time.time() - start_time
    
    if result['success']:
        logger.info(
            "Environmental data processing successful",
            processing_time=f"{processing_time:.3f}s",
            engine_type=result['engine_type']
        )
        
        # Validate results
        stats = result['statistics']
        forecast = result['forecast']
        
        # Check statistics
        temp_stats = stats.get('temperature', {})
        humidity_stats = stats.get('humidity', {})
        
        logger.info(
            "Processing statistics",
            temp_mean=temp_stats.get('mean', 0),
            temp_std=temp_stats.get('std', 0),
            humidity_mean=humidity_stats.get('mean', 0),
            humidity_std=humidity_stats.get('std', 0)
        )
        
        # Check forecast
        if forecast and 'timestamps' in forecast and len(forecast['timestamps']) > 0:
            logger.info(
                "Forecast generated",
                forecast_points=len(forecast['timestamps']),
                forecast_horizon=f"{len(forecast['timestamps'])} hours"
            )
        else:
            logger.warning("No forecast generated")
        
        return {
            'success': True,
            'processing_time': processing_time,
            'statistics': stats,
            'forecast': forecast,
            'engine_type': result['engine_type']
        }
    else:
        logger.error("Environmental data processing failed", error=result['error'])
        return {'success': False, 'error': result['error']}


def test_simulink_model_creation(manager: MATLABEngineManager) -> dict:
    """Test Simulink model creation and basic functionality."""
    
    logger.info("Testing Simulink model creation")
    
    try:
        # Call model creation function
        result = manager.call_function(
            'create_predictive_maintenance_model',
            nargout=1
        )
        
        if result.success:
            model_name = result.data
            logger.info(
                "Simulink model created successfully",
                model_name=model_name,
                creation_time=f"{result.execution_time:.3f}s"
            )
            
            # Test model compilation
            compile_result = manager.call_function(
                'eval',
                f"try; {model_name}([],[],[],'compile'); {model_name}([],[],[],'term'); success=1; catch; success=0; end",
                nargout=0
            )
            
            success_check = manager.get_variable('success')
            
            if success_check == 1:
                logger.info("Simulink model compilation successful")
                return {
                    'success': True,
                    'model_name': model_name,
                    'creation_time': result.execution_time
                }
            else:
                logger.warning("Simulink model compilation failed")
                return {
                    'success': False,
                    'error': 'Model compilation failed'
                }
        else:
            logger.error("Simulink model creation failed", error=result.error_message)
            return {'success': False, 'error': result.error_message}
            
    except Exception as e:
        logger.error("Simulink model creation exception", error=str(e))
        return {'success': False, 'error': str(e)}


def test_simulink_simulation(manager: MATLABEngineManager) -> dict:
    """Test Simulink model simulation."""
    
    logger.info("Testing Simulink model simulation")
    
    # Generate test data
    t_vib, vibration = generate_synthetic_vibration_data(
        duration_seconds=100.0,
        sample_rate=100.0,
        add_anomalies=True
    )
    
    # Generate environmental data (lower sample rate)
    t_env = np.linspace(0, 100, 101)  # 1 Hz
    temperature = 25 + 5 * np.sin(2 * np.pi * t_env / 50) + np.random.normal(0, 1, len(t_env))
    humidity = 60 - 10 * np.sin(2 * np.pi * t_env / 50) + np.random.normal(0, 2, len(t_env))
    
    # Create Simulink runner
    runner = SimulinkModelRunner(manager)
    
    # Run simulation
    start_time = time.time()
    result = runner.run_predictive_maintenance_model(
        vibration, temperature, humidity, simulation_time=100.0
    )
    simulation_time = time.time() - start_time
    
    if result['success']:
        logger.info(
            "Simulink simulation successful",
            simulation_time=f"{simulation_time:.3f}s",
            engine_type=result['engine_type']
        )
        
        return {
            'success': True,
            'simulation_time': simulation_time,
            'health_score_log': result.get('health_score_log'),
            'anomaly_log': result.get('anomaly_log'),
            'engine_type': result['engine_type']
        }
    else:
        logger.error("Simulink simulation failed", error=result['error'])
        return {'success': False, 'error': result['error']}


def benchmark_performance(
    manager: MATLABEngineManager,
    num_iterations: int = 10
) -> dict:
    """Benchmark MATLAB/Octave performance."""
    
    logger.info("Starting performance benchmark", iterations=num_iterations)
    
    # Test different operation types
    benchmarks = {}
    
    # Basic arithmetic
    times = []
    for _ in range(num_iterations):
        start_time = time.time()
        result = manager.evaluate_expression("sum(1:1000)")
        times.append(time.time() - start_time)
    
    benchmarks['arithmetic'] = {
        'avg_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times)
    }
    
    # Array operations
    test_array = np.random.randn(1000)
    manager.set_variable('test_array', test_array)
    
    times = []
    for _ in range(num_iterations):
        start_time = time.time()
        result = manager.evaluate_expression("fft(test_array)")
        times.append(time.time() - start_time)
    
    benchmarks['fft'] = {
        'avg_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times)
    }
    
    # Function calls
    times = []
    for _ in range(num_iterations):
        start_time = time.time()
        result = manager.call_function('sqrt', 16, nargout=1)
        times.append(time.time() - start_time)
    
    benchmarks['function_call'] = {
        'avg_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times)
    }
    
    logger.info("Performance benchmark completed")
    
    return benchmarks


def run_complete_matlab_integration_test(
    matlab_path: str,
    prefer_matlab: bool = True,
    run_simulink: bool = True
) -> dict:
    """Run complete MATLAB/Simulink integration test."""
    
    logger.info(
        "Starting complete MATLAB integration test",
        matlab_path=matlab_path,
        prefer_matlab=prefer_matlab,
        run_simulink=run_simulink
    )
    
    results = {}
    
    # Step 1: Initialize MATLAB Engine
    logger.info("Step 1: Initializing MATLAB Engine")
    manager = test_matlab_engine_initialization(matlab_path, prefer_matlab)
    
    if manager is None:
        return {'success': False, 'error': 'Failed to initialize computation engine'}
    
    results['engine_initialization'] = {
        'success': True,
        'engine_type': manager.engine_type
    }
    
    try:
        # Step 2: Test environmental data processing
        logger.info("Step 2: Testing environmental data processing")
        env_result = test_environmental_data_processing(manager, duration_hours=48.0)
        results['environmental_processing'] = env_result
        
        # Step 3: Test Simulink model creation (if MATLAB available)
        if run_simulink and manager.engine_type == "matlab":
            logger.info("Step 3: Testing Simulink model creation")
            model_result = test_simulink_model_creation(manager)
            results['simulink_model_creation'] = model_result
            
            # Step 4: Test Simulink simulation
            if model_result['success']:
                logger.info("Step 4: Testing Simulink simulation")
                sim_result = test_simulink_simulation(manager)
                results['simulink_simulation'] = sim_result
            else:
                logger.warning("Skipping Simulink simulation due to model creation failure")
                results['simulink_simulation'] = {'success': False, 'error': 'Model creation failed'}
        else:
            logger.info("Skipping Simulink tests (not available with Octave)")
            results['simulink_model_creation'] = {'success': False, 'error': 'Simulink not available'}
            results['simulink_simulation'] = {'success': False, 'error': 'Simulink not available'}
        
        # Step 5: Performance benchmark
        logger.info("Step 5: Running performance benchmark")
        benchmark_result = benchmark_performance(manager, num_iterations=5)
        results['performance_benchmark'] = benchmark_result
        
        # Step 6: Get final performance stats
        perf_stats = manager.get_performance_stats()
        results['final_performance_stats'] = perf_stats
        
        logger.info("Complete MATLAB integration test finished successfully")
        results['success'] = True
        
    except Exception as e:
        logger.error("Integration test failed", error=str(e), exc_info=True)
        results['success'] = False
        results['error'] = str(e)
    
    finally:
        # Cleanup
        logger.info("Shutting down computation engines")
        manager.shutdown()
    
    return results


def print_results_summary(results: dict):
    """Print a summary of integration test results."""
    
    print("\n" + "="*80)
    print("MATLAB/SIMULINK INTEGRATION TEST RESULTS")
    print("="*80)
    
    if results.get('success', False):
        print("✅ Overall Status: SUCCESS")
    else:
        print("❌ Overall Status: FAILED")
        if 'error' in results:
            print(f"   Error: {results['error']}")
    
    # Engine initialization
    if 'engine_initialization' in results:
        init = results['engine_initialization']
        if init['success']:
            print(f"✅ Engine Initialization: {init['engine_type'].upper()}")
        else:
            print("❌ Engine Initialization: FAILED")
    
    # Environmental processing
    if 'environmental_processing' in results:
        env = results['environmental_processing']
        if env['success']:
            print(f"✅ Environmental Processing: {env['processing_time']:.3f}s ({env['engine_type']})")
        else:
            print("❌ Environmental Processing: FAILED")
    
    # Simulink model creation
    if 'simulink_model_creation' in results:
        model = results['simulink_model_creation']
        if model['success']:
            print(f"✅ Simulink Model Creation: {model['creation_time']:.3f}s")
        else:
            print(f"⚠️  Simulink Model Creation: {model.get('error', 'FAILED')}")
    
    # Simulink simulation
    if 'simulink_simulation' in results:
        sim = results['simulink_simulation']
        if sim['success']:
            print(f"✅ Simulink Simulation: {sim['simulation_time']:.3f}s")
        else:
            print(f"⚠️  Simulink Simulation: {sim.get('error', 'FAILED')}")
    
    # Performance benchmark
    if 'performance_benchmark' in results:
        bench = results['performance_benchmark']
        print("📊 Performance Benchmark:")
        for op_type, metrics in bench.items():
            print(f"   {op_type}: {metrics['avg_time']*1000:.2f}ms ± {metrics['std_time']*1000:.2f}ms")
    
    # Final stats
    if 'final_performance_stats' in results:
        stats = results['final_performance_stats']
        print(f"📈 Total Function Calls: {stats['call_count']}")
        print(f"📈 Average Execution Time: {stats['avg_execution_time']*1000:.2f}ms")
    
    print("="*80)


def main():
    """Main function for MATLAB integration testing."""
    
    parser = argparse.ArgumentParser(description="MATLAB/Simulink Integration Test")
    parser.add_argument(
        "--matlab-path",
        default=str(Path(__file__).parent.parent / "matlab"),
        help="Path to MATLAB scripts directory"
    )
    parser.add_argument(
        "--prefer-octave",
        action="store_true",
        help="Prefer Octave over MATLAB"
    )
    parser.add_argument(
        "--skip-simulink",
        action="store_true",
        help="Skip Simulink tests"
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
        # Run complete integration test
        results = run_complete_matlab_integration_test(
            matlab_path=args.matlab_path,
            prefer_matlab=not args.prefer_octave,
            run_simulink=not args.skip_simulink
        )
        
        # Print summary
        print_results_summary(results)
        
        # Exit with appropriate code
        if results.get('success', False):
            sys.exit(0)
        else:
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Integration test interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error("Integration test failed", error=str(e), exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

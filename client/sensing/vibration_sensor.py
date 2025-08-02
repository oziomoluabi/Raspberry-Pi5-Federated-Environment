"""
Vibration Sensor Interface for ADXL345 Accelerometer
Raspberry Pi 5 Federated Environmental Monitoring Network
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
import structlog
import time
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import json

# Try to import hardware-specific libraries
try:
    import board
    import busio
    import adafruit_adxl34x
    HARDWARE_AVAILABLE = True
except ImportError:
    HARDWARE_AVAILABLE = False
    logger = structlog.get_logger(__name__)
    logger.warning("Hardware libraries not available, using simulation mode")

logger = structlog.get_logger(__name__)


@dataclass
class VibrationReading:
    """Data class for vibration sensor readings."""
    timestamp: float
    x_accel: float
    y_accel: float
    z_accel: float
    magnitude: float
    sample_rate: float


class ADXL345VibrationSensor:
    """Interface for ADXL345 3-axis accelerometer for vibration monitoring."""
    
    def __init__(
        self,
        sample_rate: int = 400,  # Hz
        range_setting: int = 16,  # ±16g
        buffer_size: int = 2048,
        simulation_mode: bool = None
    ):
        """Initialize ADXL345 vibration sensor."""
        
        self.sample_rate = sample_rate
        self.range_setting = range_setting
        self.buffer_size = buffer_size
        
        # Determine if we should use simulation mode
        if simulation_mode is None:
            self.simulation_mode = not HARDWARE_AVAILABLE
        else:
            self.simulation_mode = simulation_mode
        
        # Sensor state
        self.sensor = None
        self.is_collecting = False
        self.collection_thread = None
        
        # Data buffers
        self.data_buffer = deque(maxlen=buffer_size)
        self.buffer_lock = threading.Lock()
        
        # Statistics
        self.total_samples = 0
        self.start_time = None
        
        logger.info(
            "ADXL345 vibration sensor initialized",
            sample_rate=sample_rate,
            range_setting=f"±{range_setting}g",
            buffer_size=buffer_size,
            simulation_mode=self.simulation_mode
        )
    
    def initialize_sensor(self) -> bool:
        """Initialize the ADXL345 sensor hardware."""
        
        if self.simulation_mode:
            logger.info("Using simulation mode for vibration sensor")
            return True
        
        try:
            # Initialize I2C bus
            i2c = busio.I2C(board.SCL, board.SDA)
            
            # Initialize ADXL345
            self.sensor = adafruit_adxl34x.ADXL345(i2c)
            
            # Configure sensor settings
            self.sensor.range = getattr(adafruit_adxl34x.Range, f"RANGE_{self.range_setting}G")
            self.sensor.data_rate = adafruit_adxl34x.DataRate.RATE_400_HZ
            
            # Test sensor reading
            test_reading = self.sensor.acceleration
            
            logger.info(
                "ADXL345 sensor initialized successfully",
                test_reading=test_reading,
                range=f"±{self.range_setting}g"
            )
            
            return True
            
        except Exception as e:
            logger.error("Failed to initialize ADXL345 sensor", error=str(e))
            self.simulation_mode = True
            return False
    
    def generate_synthetic_vibration(self) -> Tuple[float, float, float]:
        """Generate synthetic vibration data for testing."""
        
        current_time = time.time()
        
        # Base frequencies for different axes
        freq_x = 10 + 5 * np.sin(0.1 * current_time)  # Variable frequency
        freq_y = 15 + 3 * np.cos(0.15 * current_time)
        freq_z = 8 + 2 * np.sin(0.08 * current_time)
        
        # Generate vibration patterns
        x_accel = (
            0.2 * np.sin(2 * np.pi * freq_x * current_time) +
            0.1 * np.sin(2 * np.pi * freq_x * 2 * current_time) +
            np.random.normal(0, 0.05)
        )
        
        y_accel = (
            0.15 * np.sin(2 * np.pi * freq_y * current_time) +
            0.08 * np.cos(2 * np.pi * freq_y * 1.5 * current_time) +
            np.random.normal(0, 0.04)
        )
        
        z_accel = (
            9.81 +  # Gravity component
            0.1 * np.sin(2 * np.pi * freq_z * current_time) +
            np.random.normal(0, 0.03)
        )
        
        # Occasionally add anomalous vibrations
        if np.random.random() < 0.02:  # 2% chance
            anomaly_amplitude = np.random.uniform(1.0, 3.0)
            anomaly_freq = np.random.uniform(100, 200)
            
            x_accel += anomaly_amplitude * np.sin(2 * np.pi * anomaly_freq * current_time)
            y_accel += anomaly_amplitude * np.cos(2 * np.pi * anomaly_freq * current_time)
        
        return x_accel, y_accel, z_accel
    
    def read_single_sample(self) -> VibrationReading:
        """Read a single vibration sample."""
        
        timestamp = time.time()
        
        if self.simulation_mode:
            x_accel, y_accel, z_accel = self.generate_synthetic_vibration()
        else:
            try:
                acceleration = self.sensor.acceleration
                x_accel, y_accel, z_accel = acceleration
            except Exception as e:
                logger.error("Error reading from ADXL345", error=str(e))
                # Fallback to simulation
                x_accel, y_accel, z_accel = self.generate_synthetic_vibration()
        
        # Calculate magnitude
        magnitude = np.sqrt(x_accel**2 + y_accel**2 + z_accel**2)
        
        return VibrationReading(
            timestamp=timestamp,
            x_accel=x_accel,
            y_accel=y_accel,
            z_accel=z_accel,
            magnitude=magnitude,
            sample_rate=self.sample_rate
        )
    
    def start_collection(self) -> None:
        """Start continuous vibration data collection."""
        
        if self.is_collecting:
            logger.warning("Data collection already running")
            return
        
        if not self.initialize_sensor():
            logger.error("Failed to initialize sensor, cannot start collection")
            return
        
        self.is_collecting = True
        self.start_time = time.time()
        self.total_samples = 0
        
        def collection_worker():
            """Worker thread for continuous data collection."""
            logger.info("Starting vibration data collection")
            
            target_interval = 1.0 / self.sample_rate
            next_sample_time = time.time()
            
            while self.is_collecting:
                try:
                    # Read sample
                    reading = self.read_single_sample()
                    
                    # Add to buffer
                    with self.buffer_lock:
                        self.data_buffer.append(reading)
                        self.total_samples += 1
                    
                    # Maintain sample rate
                    next_sample_time += target_interval
                    sleep_time = next_sample_time - time.time()
                    
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    elif sleep_time < -target_interval:
                        # If we're falling behind, reset timing
                        next_sample_time = time.time()
                        logger.warning("Sample rate falling behind target")
                
                except Exception as e:
                    logger.error("Error in collection worker", error=str(e))
                    time.sleep(0.01)  # Brief pause on error
            
            logger.info("Vibration data collection stopped")
        
        # Start collection thread
        self.collection_thread = threading.Thread(
            target=collection_worker,
            name="VibrationCollection",
            daemon=True
        )
        self.collection_thread.start()
        
        logger.info("Vibration data collection started")
    
    def stop_collection(self) -> None:
        """Stop continuous data collection."""
        
        if not self.is_collecting:
            return
        
        logger.info("Stopping vibration data collection")
        self.is_collecting = False
        
        if self.collection_thread:
            self.collection_thread.join(timeout=2.0)
        
        logger.info("Vibration data collection stopped")
    
    def get_recent_data(
        self,
        num_samples: Optional[int] = None,
        time_window: Optional[float] = None
    ) -> List[VibrationReading]:
        """Get recent vibration data from buffer."""
        
        with self.buffer_lock:
            if not self.data_buffer:
                return []
            
            if num_samples is not None:
                # Get last N samples
                start_idx = max(0, len(self.data_buffer) - num_samples)
                return list(self.data_buffer)[start_idx:]
            
            elif time_window is not None:
                # Get samples within time window
                current_time = time.time()
                cutoff_time = current_time - time_window
                
                recent_data = []
                for reading in reversed(self.data_buffer):
                    if reading.timestamp >= cutoff_time:
                        recent_data.append(reading)
                    else:
                        break
                
                return list(reversed(recent_data))
            
            else:
                # Return all data
                return list(self.data_buffer)
    
    def get_vibration_features(
        self,
        data: Optional[List[VibrationReading]] = None,
        window_size: int = 128
    ) -> np.ndarray:
        """Extract vibration features for ML processing."""
        
        if data is None:
            data = self.get_recent_data(num_samples=window_size)
        
        if len(data) < window_size:
            logger.warning(
                "Insufficient data for feature extraction",
                available=len(data),
                required=window_size
            )
            # Pad with zeros if insufficient data
            features = np.zeros(window_size)
            if data:
                magnitudes = [reading.magnitude for reading in data]
                features[:len(magnitudes)] = magnitudes
            return features
        
        # Extract magnitude values
        magnitudes = np.array([reading.magnitude for reading in data[-window_size:]])
        
        # Apply basic preprocessing
        # Remove DC component
        magnitudes = magnitudes - np.mean(magnitudes)
        
        # Apply window function to reduce spectral leakage
        window = np.hanning(len(magnitudes))
        magnitudes = magnitudes * window
        
        return magnitudes
    
    def get_statistics(self) -> Dict:
        """Get collection statistics."""
        
        with self.buffer_lock:
            buffer_size = len(self.data_buffer)
            
            if buffer_size > 0:
                recent_data = list(self.data_buffer)[-100:]  # Last 100 samples
                magnitudes = [r.magnitude for r in recent_data]
                
                stats = {
                    'total_samples': self.total_samples,
                    'buffer_size': buffer_size,
                    'buffer_utilization': buffer_size / self.buffer_size,
                    'is_collecting': self.is_collecting,
                    'simulation_mode': self.simulation_mode,
                    'recent_magnitude_mean': np.mean(magnitudes),
                    'recent_magnitude_std': np.std(magnitudes),
                    'recent_magnitude_max': np.max(magnitudes),
                    'sample_rate': self.sample_rate
                }
                
                if self.start_time:
                    elapsed_time = time.time() - self.start_time
                    stats['elapsed_time_s'] = elapsed_time
                    stats['actual_sample_rate'] = self.total_samples / elapsed_time if elapsed_time > 0 else 0
                
            else:
                stats = {
                    'total_samples': self.total_samples,
                    'buffer_size': 0,
                    'buffer_utilization': 0.0,
                    'is_collecting': self.is_collecting,
                    'simulation_mode': self.simulation_mode,
                    'sample_rate': self.sample_rate
                }
        
        return stats
    
    def save_data_to_file(
        self,
        filepath: str,
        num_samples: Optional[int] = None
    ) -> None:
        """Save vibration data to file."""
        
        data = self.get_recent_data(num_samples=num_samples)
        
        if not data:
            logger.warning("No data to save")
            return
        
        # Convert to dictionary format
        data_dict = {
            'metadata': {
                'sample_rate': self.sample_rate,
                'range_setting': self.range_setting,
                'simulation_mode': self.simulation_mode,
                'num_samples': len(data),
                'start_timestamp': data[0].timestamp,
                'end_timestamp': data[-1].timestamp
            },
            'data': [
                {
                    'timestamp': reading.timestamp,
                    'x_accel': reading.x_accel,
                    'y_accel': reading.y_accel,
                    'z_accel': reading.z_accel,
                    'magnitude': reading.magnitude
                }
                for reading in data
            ]
        }
        
        # Save to file
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(data_dict, f, indent=2)
        
        logger.info(
            "Vibration data saved",
            filepath=str(filepath),
            samples=len(data)
        )


class VibrationMonitor:
    """High-level vibration monitoring system."""
    
    def __init__(
        self,
        sensor_config: Optional[Dict] = None,
        anomaly_callback: Optional[Callable] = None
    ):
        """Initialize vibration monitor."""
        
        self.sensor_config = sensor_config or {}
        self.anomaly_callback = anomaly_callback
        
        # Initialize sensor
        self.sensor = ADXL345VibrationSensor(**self.sensor_config)
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread = None
        
        logger.info("Vibration monitor initialized")
    
    def start_monitoring(
        self,
        check_interval: float = 1.0,
        window_size: int = 128
    ) -> None:
        """Start vibration monitoring with anomaly detection."""
        
        if self.is_monitoring:
            logger.warning("Monitoring already running")
            return
        
        # Start sensor collection
        self.sensor.start_collection()
        
        self.is_monitoring = True
        
        def monitoring_worker():
            """Worker thread for vibration monitoring."""
            logger.info("Starting vibration monitoring")
            
            while self.is_monitoring:
                try:
                    # Get recent vibration features
                    features = self.sensor.get_vibration_features(window_size=window_size)
                    
                    # Simple anomaly detection (can be replaced with ML model)
                    magnitude_std = np.std(features)
                    is_anomaly = magnitude_std > 0.5  # Simple threshold
                    
                    if is_anomaly and self.anomaly_callback:
                        self.anomaly_callback({
                            'timestamp': time.time(),
                            'magnitude_std': magnitude_std,
                            'features': features.tolist()
                        })
                    
                    time.sleep(check_interval)
                
                except Exception as e:
                    logger.error("Error in monitoring worker", error=str(e))
                    time.sleep(1.0)
            
            logger.info("Vibration monitoring stopped")
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(
            target=monitoring_worker,
            name="VibrationMonitor",
            daemon=True
        )
        self.monitor_thread.start()
        
        logger.info("Vibration monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop vibration monitoring."""
        
        if not self.is_monitoring:
            return
        
        logger.info("Stopping vibration monitoring")
        
        self.is_monitoring = False
        self.sensor.stop_collection()
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        
        logger.info("Vibration monitoring stopped")
    
    def get_status(self) -> Dict:
        """Get monitoring status."""
        
        sensor_stats = self.sensor.get_statistics()
        
        return {
            'is_monitoring': self.is_monitoring,
            'sensor_stats': sensor_stats
        }


if __name__ == "__main__":
    # Example usage
    def anomaly_detected(data):
        print(f"Anomaly detected at {time.time()}: {data['magnitude_std']:.3f}")
    
    # Create monitor
    monitor = VibrationMonitor(
        sensor_config={'sample_rate': 400, 'simulation_mode': True},
        anomaly_callback=anomaly_detected
    )
    
    # Start monitoring
    monitor.start_monitoring()
    
    try:
        time.sleep(10)  # Monitor for 10 seconds
    except KeyboardInterrupt:
        pass
    finally:
        monitor.stop_monitoring()

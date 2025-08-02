"""
Sensor Manager for Raspberry Pi 5
Handles Sense HAT and ADXL345 sensor data collection
"""

import time
import numpy as np
import structlog
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

try:
    from sense_hat import SenseHat
    SENSE_HAT_AVAILABLE = True
except ImportError:
    SENSE_HAT_AVAILABLE = False
    structlog.get_logger(__name__).warning("Sense HAT not available, using simulation")

try:
    import board
    import busio
    import adafruit_adxl34x
    ADXL345_AVAILABLE = True
except ImportError:
    ADXL345_AVAILABLE = False
    structlog.get_logger(__name__).warning("ADXL345 not available, using simulation")

logger = structlog.get_logger(__name__)


@dataclass
class SensorReading:
    """Data class for sensor readings."""
    timestamp: datetime
    temperature: float
    humidity: float
    pressure: float
    acceleration_x: float
    acceleration_y: float
    acceleration_z: float


class SensorManager:
    """Manages all sensors on the Raspberry Pi."""
    
    def __init__(self, simulation_mode: bool = None):
        """
        Initialize sensor manager.
        
        Args:
            simulation_mode: Force simulation mode if True, auto-detect if None
        """
        self.simulation_mode = simulation_mode
        if simulation_mode is None:
            self.simulation_mode = not (SENSE_HAT_AVAILABLE and ADXL345_AVAILABLE)
        
        self.sense_hat = None
        self.adxl345 = None
        
        self._initialize_sensors()
        logger.info(
            "Sensor manager initialized",
            simulation_mode=self.simulation_mode,
            sense_hat_available=SENSE_HAT_AVAILABLE,
            adxl345_available=ADXL345_AVAILABLE
        )
    
    def _initialize_sensors(self):
        """Initialize hardware sensors if available."""
        if not self.simulation_mode:
            try:
                if SENSE_HAT_AVAILABLE:
                    self.sense_hat = SenseHat()
                    logger.info("Sense HAT initialized")
                
                if ADXL345_AVAILABLE:
                    i2c = busio.I2C(board.SCL, board.SDA)
                    self.adxl345 = adafruit_adxl34x.ADXL345(i2c)
                    logger.info("ADXL345 initialized")
                    
            except Exception as e:
                logger.warning("Failed to initialize hardware sensors", error=str(e))
                self.simulation_mode = True
    
    def read_environmental_data(self) -> Tuple[float, float, float]:
        """
        Read environmental data from Sense HAT.
        
        Returns:
            Tuple of (temperature, humidity, pressure)
        """
        if self.simulation_mode or self.sense_hat is None:
            # Simulate realistic environmental data
            base_temp = 22.0 + np.random.normal(0, 2)
            base_humidity = 45.0 + np.random.normal(0, 5)
            base_pressure = 1013.25 + np.random.normal(0, 10)
            return base_temp, base_humidity, base_pressure
        
        try:
            temperature = self.sense_hat.get_temperature()
            humidity = self.sense_hat.get_humidity()
            pressure = self.sense_hat.get_pressure()
            return temperature, humidity, pressure
        except Exception as e:
            logger.error("Failed to read environmental data", error=str(e))
            return 0.0, 0.0, 0.0
    
    def read_acceleration_data(self) -> Tuple[float, float, float]:
        """
        Read acceleration data from ADXL345.
        
        Returns:
            Tuple of (x, y, z) acceleration in m/s²
        """
        if self.simulation_mode or self.adxl345 is None:
            # Simulate vibration data with some noise
            base_x = np.random.normal(0, 0.1)
            base_y = np.random.normal(0, 0.1)
            base_z = 9.81 + np.random.normal(0, 0.2)  # Gravity + noise
            return base_x, base_y, base_z
        
        try:
            x, y, z = self.adxl345.acceleration
            return x, y, z
        except Exception as e:
            logger.error("Failed to read acceleration data", error=str(e))
            return 0.0, 0.0, 0.0
    
    def get_sensor_reading(self) -> SensorReading:
        """Get a complete sensor reading."""
        temp, humidity, pressure = self.read_environmental_data()
        acc_x, acc_y, acc_z = self.read_acceleration_data()
        
        return SensorReading(
            timestamp=datetime.now(),
            temperature=temp,
            humidity=humidity,
            pressure=pressure,
            acceleration_x=acc_x,
            acceleration_y=acc_y,
            acceleration_z=acc_z
        )
    
    def collect_batch_data(
        self,
        duration_seconds: int = 60,
        sample_rate_hz: float = 1.0
    ) -> List[SensorReading]:
        """
        Collect a batch of sensor readings over a specified duration.
        
        Args:
            duration_seconds: How long to collect data
            sample_rate_hz: Sampling rate in Hz
            
        Returns:
            List of sensor readings
        """
        readings = []
        interval = 1.0 / sample_rate_hz
        end_time = time.time() + duration_seconds
        
        logger.info(
            "Starting batch data collection",
            duration_seconds=duration_seconds,
            sample_rate_hz=sample_rate_hz
        )
        
        while time.time() < end_time:
            reading = self.get_sensor_reading()
            readings.append(reading)
            time.sleep(interval)
        
        logger.info("Batch data collection completed", num_readings=len(readings))
        return readings
    
    def get_environmental_history(
        self,
        hours: int = 24
    ) -> np.ndarray:
        """
        Get historical environmental data for training.
        
        Args:
            hours: Number of hours of history to generate/retrieve
            
        Returns:
            Array of shape (hours, 2) with temperature and humidity data
        """
        # For now, simulate historical data
        # In a real implementation, this would read from a database
        data = []
        for i in range(hours):
            temp, humidity, _ = self.read_environmental_data()
            data.append([temp, humidity])
        
        return np.array(data)
    
    def cleanup(self):
        """Clean up sensor resources."""
        logger.info("Cleaning up sensor manager")
        # Add any necessary cleanup code here

"""
Unit tests for SensorManager
"""

import pytest
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch

from client.sensing.sensor_manager import SensorManager, SensorReading


class TestSensorManager:
    """Test cases for SensorManager class."""
    
    def test_init_simulation_mode(self):
        """Test initialization in simulation mode."""
        sensor_manager = SensorManager(simulation_mode=True)
        assert sensor_manager.simulation_mode is True
        assert sensor_manager.sense_hat is None
        assert sensor_manager.adxl345 is None
    
    def test_read_environmental_data_simulation(self):
        """Test reading environmental data in simulation mode."""
        sensor_manager = SensorManager(simulation_mode=True)
        temp, humidity, pressure = sensor_manager.read_environmental_data()
        
        # Check that values are within reasonable ranges
        assert 15.0 <= temp <= 35.0  # Temperature in Celsius
        assert 20.0 <= humidity <= 80.0  # Humidity percentage
        assert 950.0 <= pressure <= 1050.0  # Pressure in hPa
    
    def test_read_acceleration_data_simulation(self):
        """Test reading acceleration data in simulation mode."""
        sensor_manager = SensorManager(simulation_mode=True)
        acc_x, acc_y, acc_z = sensor_manager.read_acceleration_data()
        
        # Check that values are within reasonable ranges
        assert -2.0 <= acc_x <= 2.0  # X acceleration
        assert -2.0 <= acc_y <= 2.0  # Y acceleration
        assert 8.0 <= acc_z <= 12.0  # Z acceleration (should be close to gravity)
    
    def test_get_sensor_reading(self):
        """Test getting a complete sensor reading."""
        sensor_manager = SensorManager(simulation_mode=True)
        reading = sensor_manager.get_sensor_reading()
        
        assert isinstance(reading, SensorReading)
        assert isinstance(reading.timestamp, datetime)
        assert isinstance(reading.temperature, float)
        assert isinstance(reading.humidity, float)
        assert isinstance(reading.pressure, float)
        assert isinstance(reading.acceleration_x, float)
        assert isinstance(reading.acceleration_y, float)
        assert isinstance(reading.acceleration_z, float)
    
    def test_collect_batch_data(self):
        """Test collecting batch data."""
        sensor_manager = SensorManager(simulation_mode=True)
        
        # Collect data for 2 seconds at 2 Hz (should get ~4 readings)
        readings = sensor_manager.collect_batch_data(
            duration_seconds=2,
            sample_rate_hz=2.0
        )
        
        assert len(readings) >= 3  # Allow for timing variations
        assert len(readings) <= 5
        assert all(isinstance(r, SensorReading) for r in readings)
    
    def test_get_environmental_history(self):
        """Test getting environmental history."""
        sensor_manager = SensorManager(simulation_mode=True)
        history = sensor_manager.get_environmental_history(hours=24)
        
        assert history.shape == (24, 2)  # 24 hours, 2 features
        assert np.all(np.isfinite(history))  # No NaN or inf values
    
    def test_cleanup(self):
        """Test cleanup method."""
        sensor_manager = SensorManager(simulation_mode=True)
        # Should not raise any exceptions
        sensor_manager.cleanup()


class TestSensorReading:
    """Test cases for SensorReading dataclass."""
    
    def test_sensor_reading_creation(self):
        """Test creating a SensorReading instance."""
        timestamp = datetime.now()
        reading = SensorReading(
            timestamp=timestamp,
            temperature=22.5,
            humidity=45.0,
            pressure=1013.25,
            acceleration_x=0.1,
            acceleration_y=-0.05,
            acceleration_z=9.81
        )
        
        assert reading.timestamp == timestamp
        assert reading.temperature == 22.5
        assert reading.humidity == 45.0
        assert reading.pressure == 1013.25
        assert reading.acceleration_x == 0.1
        assert reading.acceleration_y == -0.05
        assert reading.acceleration_z == 9.81


@pytest.mark.integration
class TestSensorManagerIntegration:
    """Integration tests for SensorManager."""
    
    def test_continuous_data_collection(self):
        """Test continuous data collection over a longer period."""
        sensor_manager = SensorManager(simulation_mode=True)
        
        # Collect data for 5 seconds
        readings = sensor_manager.collect_batch_data(
            duration_seconds=5,
            sample_rate_hz=1.0
        )
        
        # Verify data consistency
        temperatures = [r.temperature for r in readings]
        humidities = [r.humidity for r in readings]
        
        # Check that values don't change too drastically between readings
        temp_diffs = [abs(temperatures[i] - temperatures[i-1]) 
                     for i in range(1, len(temperatures))]
        humidity_diffs = [abs(humidities[i] - humidities[i-1]) 
                         for i in range(1, len(humidities))]
        
        # Temperature shouldn't change more than 5°C between readings
        assert all(diff < 5.0 for diff in temp_diffs)
        # Humidity shouldn't change more than 10% between readings
        assert all(diff < 10.0 for diff in humidity_diffs)

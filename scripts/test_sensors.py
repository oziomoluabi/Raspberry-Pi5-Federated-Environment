#!/usr/bin/env python3
"""
Raspberry Pi 5 Sensor Connectivity Test
Sprint 7: Pilot Deployment & Validation

Tests connectivity and basic functionality of:
- Sense HAT (temperature, humidity, pressure, IMU)
- ADXL345 accelerometer
"""

import sys
import time
import json
import logging
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

def test_sense_hat():
    """Test Sense HAT connectivity and sensors."""
    try:
        from sense_hat import SenseHat
        
        logger.info("Testing Sense HAT connectivity...")
        sense = SenseHat()
        
        # Test environmental sensors
        temperature = sense.get_temperature()
        humidity = sense.get_humidity()
        pressure = sense.get_pressure()
        
        logger.info(f"Sense HAT environmental readings:")
        logger.info(f"  Temperature: {temperature:.2f}°C")
        logger.info(f"  Humidity: {humidity:.2f}%")
        logger.info(f"  Pressure: {pressure:.2f} mbar")
        
        # Test IMU sensors
        orientation = sense.get_orientation()
        acceleration = sense.get_accelerometer_raw()
        gyroscope = sense.get_gyroscope_raw()
        compass = sense.get_compass_raw()
        
        logger.info(f"Sense HAT IMU readings:")
        logger.info(f"  Orientation: pitch={orientation['pitch']:.2f}, roll={orientation['roll']:.2f}, yaw={orientation['yaw']:.2f}")
        logger.info(f"  Acceleration: x={acceleration['x']:.3f}, y={acceleration['y']:.3f}, z={acceleration['z']:.3f}")
        
        # Test LED matrix
        sense.set_pixel(0, 0, (0, 255, 0))  # Green pixel
        time.sleep(0.5)
        sense.clear()
        
        logger.info("Sense HAT test completed successfully")
        return {
            'status': 'success',
            'environmental': {
                'temperature': temperature,
                'humidity': humidity,
                'pressure': pressure
            },
            'imu': {
                'orientation': orientation,
                'acceleration': acceleration,
                'gyroscope': gyroscope,
                'compass': compass
            }
        }
        
    except ImportError as e:
        logger.error(f"Sense HAT library not available: {e}")
        return {'status': 'error', 'error': f'Import error: {e}'}
    except Exception as e:
        logger.error(f"Sense HAT test failed: {e}")
        return {'status': 'error', 'error': str(e)}

def test_adxl345():
    """Test ADXL345 accelerometer connectivity."""
    try:
        import board
        import busio
        import adafruit_adxl34x
        
        logger.info("Testing ADXL345 accelerometer connectivity...")
        
        # Initialize I2C bus
        i2c = busio.I2C(board.SCL, board.SDA)
        accelerometer = adafruit_adxl34x.ADXL345(i2c)
        
        # Configure accelerometer
        accelerometer.range = adafruit_adxl34x.Range.RANGE_16_G
        accelerometer.data_rate = adafruit_adxl34x.DataRate.RATE_100_HZ
        
        # Take multiple readings
        readings = []
        for i in range(10):
            x, y, z = accelerometer.acceleration
            readings.append({'x': x, 'y': y, 'z': z})
            time.sleep(0.01)
        
        # Calculate average
        avg_x = sum(r['x'] for r in readings) / len(readings)
        avg_y = sum(r['y'] for r in readings) / len(readings)
        avg_z = sum(r['z'] for r in readings) / len(readings)
        
        logger.info(f"ADXL345 average acceleration: x={avg_x:.3f}, y={avg_y:.3f}, z={avg_z:.3f} m/s²")
        
        # Check if readings are reasonable (gravity should be ~9.8 m/s²)
        magnitude = (avg_x**2 + avg_y**2 + avg_z**2)**0.5
        logger.info(f"Acceleration magnitude: {magnitude:.3f} m/s² (expected ~9.8)")
        
        if 8.0 < magnitude < 12.0:  # Reasonable range for gravity
            logger.info("ADXL345 test completed successfully")
            return {
                'status': 'success',
                'readings': readings,
                'average': {'x': avg_x, 'y': avg_y, 'z': avg_z},
                'magnitude': magnitude
            }
        else:
            logger.warning(f"ADXL345 readings may be incorrect (magnitude: {magnitude:.3f})")
            return {
                'status': 'warning',
                'readings': readings,
                'average': {'x': avg_x, 'y': avg_y, 'z': avg_z},
                'magnitude': magnitude,
                'warning': 'Acceleration magnitude outside expected range'
            }
            
    except ImportError as e:
        logger.error(f"ADXL345 library not available: {e}")
        return {'status': 'error', 'error': f'Import error: {e}'}
    except Exception as e:
        logger.error(f"ADXL345 test failed: {e}")
        return {'status': 'error', 'error': str(e)}

def test_i2c_devices():
    """Scan for I2C devices."""
    try:
        import subprocess
        
        logger.info("Scanning for I2C devices...")
        result = subprocess.run(['i2cdetect', '-y', '1'], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            logger.info("I2C scan completed:")
            for line in result.stdout.split('\n'):
                if line.strip():
                    logger.info(f"  {line}")
            
            # Look for expected devices
            output = result.stdout
            sense_hat_found = '5c' in output  # Sense HAT typically at 0x5c
            adxl345_found = '53' in output    # ADXL345 typically at 0x53
            
            return {
                'status': 'success',
                'scan_output': result.stdout,
                'devices_found': {
                    'sense_hat': sense_hat_found,
                    'adxl345': adxl345_found
                }
            }
        else:
            logger.error(f"I2C scan failed: {result.stderr}")
            return {'status': 'error', 'error': result.stderr}
            
    except subprocess.TimeoutExpired:
        logger.error("I2C scan timed out")
        return {'status': 'error', 'error': 'Scan timeout'}
    except Exception as e:
        logger.error(f"I2C scan failed: {e}")
        return {'status': 'error', 'error': str(e)}

def test_system_info():
    """Gather system information."""
    try:
        import platform
        import psutil
        
        logger.info("Gathering system information...")
        
        # System info
        system_info = {
            'platform': platform.platform(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total,
            'disk_usage': psutil.disk_usage('/').percent
        }
        
        # Temperature (Pi-specific)
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                temp_raw = int(f.read().strip())
                system_info['cpu_temperature'] = temp_raw / 1000.0
        except:
            system_info['cpu_temperature'] = None
        
        logger.info(f"System: {system_info['platform']}")
        logger.info(f"CPU: {system_info['cpu_count']} cores, {system_info['processor']}")
        logger.info(f"Memory: {system_info['memory_total'] / (1024**3):.1f} GB")
        if system_info['cpu_temperature']:
            logger.info(f"CPU Temperature: {system_info['cpu_temperature']:.1f}°C")
        
        return {'status': 'success', 'system_info': system_info}
        
    except Exception as e:
        logger.error(f"System info gathering failed: {e}")
        return {'status': 'error', 'error': str(e)}

def main():
    """Main test routine."""
    logger.info("Starting Raspberry Pi 5 sensor connectivity test")
    logger.info("=" * 60)
    
    test_results = {
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'test_type': 'sensor_connectivity',
        'results': {}
    }
    
    # Test system info
    logger.info("Testing system information...")
    test_results['results']['system'] = test_system_info()
    
    # Test I2C devices
    logger.info("Testing I2C connectivity...")
    test_results['results']['i2c'] = test_i2c_devices()
    
    # Test Sense HAT
    logger.info("Testing Sense HAT...")
    test_results['results']['sense_hat'] = test_sense_hat()
    
    # Test ADXL345
    logger.info("Testing ADXL345...")
    test_results['results']['adxl345'] = test_adxl345()
    
    # Summary
    logger.info("=" * 60)
    logger.info("Test Summary:")
    
    success_count = 0
    total_tests = 0
    
    for test_name, result in test_results['results'].items():
        total_tests += 1
        status = result.get('status', 'unknown')
        
        if status == 'success':
            success_count += 1
            logger.info(f"  {test_name}: ✓ PASS")
        elif status == 'warning':
            logger.info(f"  {test_name}: ⚠ WARN")
        else:
            logger.info(f"  {test_name}: ✗ FAIL - {result.get('error', 'Unknown error')}")
    
    logger.info(f"Overall: {success_count}/{total_tests} tests passed")
    
    # Save results to file
    results_file = Path(__file__).parent.parent / 'logs' / 'sensor_test_results.json'
    results_file.parent.mkdir(exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(test_results, f, indent=2)
    
    logger.info(f"Test results saved to: {results_file}")
    
    # Exit with appropriate code
    if success_count == total_tests:
        logger.info("All tests passed successfully!")
        sys.exit(0)
    elif success_count > 0:
        logger.warning("Some tests failed, but basic functionality is available")
        sys.exit(1)
    else:
        logger.error("All tests failed - hardware may not be properly connected")
        sys.exit(2)

if __name__ == "__main__":
    main()

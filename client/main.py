#!/usr/bin/env python3
"""
Federated Learning Client for Raspberry Pi 5
Environmental Monitoring and TinyML Predictive Maintenance
"""

import logging
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import flwr as fl
import structlog
from client.sensing.sensor_manager import SensorManager
from client.training.federated_client import EnvironmentalClient
from client.training.autoencoder import AutoencoderManager

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


def main():
    """Main entry point for the federated learning client."""
    logger.info("Starting Federated Environmental Monitoring Client")
    
    try:
        # Initialize sensor manager
        sensor_manager = SensorManager()
        logger.info("Sensor manager initialized")
        
        # Initialize autoencoder for predictive maintenance
        autoencoder_manager = AutoencoderManager()
        logger.info("Autoencoder manager initialized")
        
        # Create federated learning client
        client = EnvironmentalClient(
            sensor_manager=sensor_manager,
            autoencoder_manager=autoencoder_manager
        )
        
        # Start the Flower client
        fl.client.start_numpy_client(
            server_address="localhost:8080",
            client=client
        )
        
    except KeyboardInterrupt:
        logger.info("Client shutdown requested by user")
    except Exception as e:
        logger.error("Client error", error=str(e), exc_info=True)
        sys.exit(1)
    finally:
        logger.info("Federated client stopped")


if __name__ == "__main__":
    main()

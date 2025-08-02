#!/usr/bin/env python3
"""
Federated Learning Server for Environmental Monitoring
Raspberry Pi 5 Federated Environmental Monitoring Network
"""

import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import flwr as fl
import structlog
from server.aggregation.federated_server import FederatedEnvironmentalServer
from server.models.lstm_model import create_lstm_model

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
    """Main entry point for the federated learning server."""
    logger.info("Starting Federated Environmental Monitoring Server")
    
    try:
        # Create the LSTM model architecture
        model = create_lstm_model(
            input_shape=(24, 2),  # 24 hours, 2 features (temp, humidity)
            lstm_units=64,
            dropout_rate=0.2
        )
        
        # Initialize the federated server
        server = FederatedEnvironmentalServer(model)
        
        # Start the Flower server
        fl.server.start_server(
            server_address="0.0.0.0:8080",
            config=fl.server.ServerConfig(num_rounds=10),
            strategy=server.get_strategy()
        )
        
    except KeyboardInterrupt:
        logger.info("Server shutdown requested by user")
    except Exception as e:
        logger.error("Server error", error=str(e), exc_info=True)
        sys.exit(1)
    finally:
        logger.info("Federated server stopped")


if __name__ == "__main__":
    main()

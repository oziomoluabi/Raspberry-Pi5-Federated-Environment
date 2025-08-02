#!/usr/bin/env python3
"""
Federated Learning Client Entry Point
Raspberry Pi 5 Federated Environmental Monitoring Network
"""

import argparse
import structlog
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from client.training.federated_client import start_federated_client

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
    """Main entry point for federated learning client."""
    
    parser = argparse.ArgumentParser(
        description="Federated Learning Client for Environmental Monitoring"
    )
    parser.add_argument(
        "--server", 
        default="localhost:8080",
        help="Server address (default: localhost:8080)"
    )
    parser.add_argument(
        "--client-id", 
        default="client_1",
        help="Client identifier (default: client_1)"
    )
    parser.add_argument(
        "--config", 
        help="Path to configuration file"
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
    
    logger.info(
        "Starting Federated Learning Client",
        server=args.server,
        client_id=args.client_id,
        config=args.config
    )
    
    try:
        start_federated_client(
            server_address=args.server,
            client_id=args.client_id,
            config_path=args.config
        )
        logger.info("Client completed successfully")
    except KeyboardInterrupt:
        logger.info("Client interrupted by user")
    except Exception as e:
        logger.error("Client failed", error=str(e), exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

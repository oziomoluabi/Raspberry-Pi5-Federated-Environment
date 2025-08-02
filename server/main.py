#!/usr/bin/env python3
"""
Federated Learning Server Entry Point
Raspberry Pi 5 Federated Environmental Monitoring Network
"""

import argparse
import structlog
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from server.aggregation.federated_server import start_federated_server

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
    """Main entry point for federated learning server."""
    
    parser = argparse.ArgumentParser(
        description="Federated Learning Server for Environmental Monitoring"
    )
    parser.add_argument(
        "--address", 
        default="0.0.0.0:8080",
        help="Server address (default: 0.0.0.0:8080)"
    )
    parser.add_argument(
        "--rounds", 
        type=int, 
        default=5,
        help="Number of federated learning rounds (default: 5)"
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
        "Starting Federated Learning Server",
        address=args.address,
        rounds=args.rounds,
        config=args.config
    )
    
    try:
        start_federated_server(
            server_address=args.address,
            config_path=args.config,
            num_rounds=args.rounds
        )
        logger.info("Server completed successfully")
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    except Exception as e:
        logger.error("Server failed", error=str(e), exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
Federated Learning Server Implementation using Flower
Raspberry Pi 5 Federated Environmental Monitoring Network
"""

import flwr as fl
import tensorflow as tf
from typing import Dict, List, Optional, Tuple, Union
import structlog
import numpy as np
from pathlib import Path
import yaml

from ..models.lstm_model import create_federated_model

logger = structlog.get_logger(__name__)


class FederatedStrategy(fl.server.strategy.FedAvg):
    """Custom federated averaging strategy for environmental monitoring."""
    
    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[callable] = None,
        on_fit_config_fn: Optional[callable] = None,
        on_evaluate_config_fn: Optional[callable] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[fl.common.Parameters] = None,
    ):
        """Initialize federated strategy with environmental monitoring parameters."""
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
        )
        
        logger.info(
            "Initialized federated strategy",
            min_clients=min_available_clients,
            fraction_fit=fraction_fit
        )
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes], BaseException]],
    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, fl.common.Scalar]]:
        """Aggregate fit results from all clients."""
        
        logger.info(
            "Aggregating fit results",
            round=server_round,
            num_results=len(results),
            num_failures=len(failures)
        )
        
        # Call parent aggregation
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        
        # Log aggregation metrics
        if aggregated_metrics:
            logger.info(
                "Aggregation complete",
                round=server_round,
                metrics=aggregated_metrics
            )
        
        return aggregated_parameters, aggregated_metrics
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, fl.common.Scalar]]:
        """Aggregate evaluation results from all clients."""
        
        logger.info(
            "Aggregating evaluation results",
            round=server_round,
            num_results=len(results),
            num_failures=len(failures)
        )
        
        # Call parent aggregation
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        
        if loss is not None:
            logger.info(
                "Evaluation aggregation complete",
                round=server_round,
                loss=loss,
                metrics=metrics
            )
        
        return loss, metrics


def get_evaluate_fn(model: tf.keras.Model, test_data: Optional[Tuple[np.ndarray, np.ndarray]] = None):
    """Return an evaluation function for server-side evaluation."""
    
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        """Evaluate global model on server-side test data."""
        
        if test_data is None:
            logger.warning("No test data available for server-side evaluation")
            return None
        
        # Set model parameters
        model.set_weights(parameters)
        
        # Evaluate model
        x_test, y_test = test_data
        loss, mae = model.evaluate(x_test, y_test, verbose=0)
        
        logger.info(
            "Server-side evaluation",
            round=server_round,
            loss=loss,
            mae=mae
        )
        
        return loss, {"mae": mae}
    
    return evaluate


def get_on_fit_config_fn():
    """Return function that configures training on client side."""
    
    def fit_config(server_round: int):
        """Return training configuration dict for each round."""
        config = {
            "batch_size": 32,
            "local_epochs": 1,
            "learning_rate": 0.001,
            "server_round": server_round,
        }
        
        # Adjust learning rate based on round
        if server_round > 10:
            config["learning_rate"] = 0.0005
        if server_round > 20:
            config["learning_rate"] = 0.0001
            
        logger.info("Fit configuration", round=server_round, config=config)
        return config
    
    return fit_config


def get_on_evaluate_config_fn():
    """Return function that configures evaluation on client side."""
    
    def evaluate_config(server_round: int):
        """Return evaluation configuration dict for each round."""
        config = {
            "batch_size": 32,
            "server_round": server_round,
        }
        
        logger.info("Evaluate configuration", round=server_round, config=config)
        return config
    
    return evaluate_config


def create_server_strategy(
    model: tf.keras.Model,
    config: Dict,
    test_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
) -> FederatedStrategy:
    """Create and configure the federated learning strategy."""
    
    # Get initial model parameters
    initial_parameters = fl.common.ndarrays_to_parameters(model.get_weights())
    
    # Create strategy
    strategy = FederatedStrategy(
        fraction_fit=config.get("fraction_fit", 1.0),
        fraction_evaluate=config.get("fraction_evaluate", 1.0),
        min_fit_clients=config.get("min_fit_clients", 2),
        min_evaluate_clients=config.get("min_evaluate_clients", 2),
        min_available_clients=config.get("min_available_clients", 2),
        evaluate_fn=get_evaluate_fn(model, test_data),
        on_fit_config_fn=get_on_fit_config_fn(),
        on_evaluate_config_fn=get_on_evaluate_config_fn(),
        accept_failures=True,
        initial_parameters=initial_parameters,
    )
    
    logger.info("Created federated strategy", config=config)
    return strategy


def start_federated_server(
    server_address: str = "0.0.0.0:8080",
    config_path: Optional[str] = None,
    num_rounds: int = 5
):
    """Start the federated learning server."""
    
    logger.info(
        "Starting federated server",
        address=server_address,
        rounds=num_rounds
    )
    
    # Load configuration
    if config_path:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {
            "fraction_fit": 1.0,
            "fraction_evaluate": 1.0,
            "min_fit_clients": 2,
            "min_evaluate_clients": 2,
            "min_available_clients": 2,
        }
    
    # Create model
    model = create_federated_model()
    logger.info("Created federated model", params=model.count_params())
    
    # Create strategy
    strategy = create_server_strategy(model, config)
    
    # Start server
    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )
    
    logger.info("Federated server completed")


if __name__ == "__main__":
    start_federated_server()

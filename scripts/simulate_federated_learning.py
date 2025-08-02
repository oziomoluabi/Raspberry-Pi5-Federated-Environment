#!/usr/bin/env python3
"""
Federated Learning Simulation Driver
Raspberry Pi 5 Federated Environmental Monitoring Network

This script simulates federated learning with multiple clients using Flower's
simulation capabilities for testing and development.
"""

import flwr as fl
import tensorflow as tf
import numpy as np
from typing import Dict, List, Tuple, Optional
import structlog
import time
import threading
import multiprocessing as mp
from pathlib import Path
import sys
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from server.models.lstm_model import create_federated_model
from client.training.federated_client import EnvironmentalClient, create_client_data

logger = structlog.get_logger(__name__)


def create_client_fn(client_id: str, total_samples: int = 800):
    """Create a client function for simulation."""
    
    def client_fn(cid: str) -> fl.client.NumPyClient:
        """Create and return a client instance."""
        
        # Create model
        model = create_federated_model()
        
        # Create client-specific data
        x_train, y_train, x_val, y_val = create_client_data(
            client_id=f"sim_client_{cid}",
            total_samples=total_samples,
            val_split=0.2,
            seed=int(cid) * 42  # Ensure different data per client
        )
        
        # Create client
        client = EnvironmentalClient(
            model=model,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            client_id=f"sim_client_{cid}"
        )
        
        return client
    
    return client_fn


def get_evaluate_fn():
    """Return evaluation function for server-side evaluation."""
    
    # Create test data
    model = create_federated_model()
    x_test, y_test = create_client_data(
        client_id="server_test",
        total_samples=200,
        val_split=0.0,
        seed=9999
    )[:2]  # Only take x_train, y_train as test data
    
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        """Evaluate global model on server test data."""
        
        # Set model parameters
        model.set_weights(parameters)
        
        # Evaluate
        loss, mae = model.evaluate(x_test, y_test, verbose=0)
        
        logger.info(
            "Server evaluation",
            round=server_round,
            loss=loss,
            mae=mae,
            test_samples=len(x_test)
        )
        
        return loss, {"mae": mae, "test_samples": len(x_test)}
    
    return evaluate


def get_fit_config(server_round: int):
    """Return training configuration for each round."""
    config = {
        "batch_size": 32,
        "local_epochs": 2,  # More epochs for simulation
        "learning_rate": 0.001,
        "server_round": server_round,
    }
    
    # Decay learning rate
    if server_round > 3:
        config["learning_rate"] = 0.0005
    if server_round > 7:
        config["learning_rate"] = 0.0001
    
    return config


def get_evaluate_config(server_round: int):
    """Return evaluation configuration for each round."""
    return {
        "batch_size": 32,
        "server_round": server_round,
    }


def run_simulation(
    num_clients: int = 3,
    num_rounds: int = 5,
    fraction_fit: float = 1.0,
    fraction_evaluate: float = 1.0,
    min_fit_clients: int = 2,
    min_evaluate_clients: int = 2,
    min_available_clients: int = 2,
    client_samples: int = 800
):
    """Run federated learning simulation."""
    
    logger.info(
        "Starting federated learning simulation",
        num_clients=num_clients,
        num_rounds=num_rounds,
        fraction_fit=fraction_fit,
        min_fit_clients=min_fit_clients,
        client_samples=client_samples
    )
    
    # Create initial model to get parameters
    model = create_federated_model()
    initial_parameters = fl.common.ndarrays_to_parameters(model.get_weights())
    
    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=min_available_clients,
        evaluate_fn=get_evaluate_fn(),
        on_fit_config_fn=get_fit_config,
        on_evaluate_config_fn=get_evaluate_config,
        initial_parameters=initial_parameters,
    )
    
    # Create client function
    client_fn = create_client_fn("simulation", client_samples)
    
    # Start simulation
    start_time = time.time()
    
    logger.info("Starting Flower simulation...")
    
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0},
    )
    
    end_time = time.time()
    total_time = end_time - start_time
    
    logger.info(
        "Simulation completed",
        total_time=f"{total_time:.2f}s",
        avg_round_time=f"{total_time/num_rounds:.2f}s",
        num_rounds=num_rounds
    )
    
    # Print results summary
    print("\n" + "="*60)
    print("FEDERATED LEARNING SIMULATION RESULTS")
    print("="*60)
    print(f"Total Rounds: {num_rounds}")
    print(f"Total Clients: {num_clients}")
    print(f"Total Time: {total_time:.2f} seconds")
    print(f"Average Round Time: {total_time/num_rounds:.2f} seconds")
    
    if history.losses_distributed:
        print(f"\nFinal Training Loss: {history.losses_distributed[-1][1]:.4f}")
    
    if history.losses_centralized:
        print(f"Final Server Loss: {history.losses_centralized[-1][1]:.4f}")
    
    if history.metrics_centralized and "mae" in history.metrics_centralized:
        final_mae = history.metrics_centralized["mae"][-1][1]
        print(f"Final Server MAE: {final_mae:.4f}")
    
    print("="*60)
    
    return history


def benchmark_performance(
    num_clients_list: List[int] = [2, 3, 5],
    num_rounds: int = 3,
    client_samples: int = 500
):
    """Benchmark federated learning performance with different client counts."""
    
    logger.info(
        "Starting performance benchmark",
        client_counts=num_clients_list,
        rounds=num_rounds,
        samples_per_client=client_samples
    )
    
    results = {}
    
    for num_clients in num_clients_list:
        logger.info(f"Benchmarking with {num_clients} clients...")
        
        start_time = time.time()
        history = run_simulation(
            num_clients=num_clients,
            num_rounds=num_rounds,
            min_fit_clients=min(2, num_clients),
            min_evaluate_clients=min(2, num_clients),
            min_available_clients=min(2, num_clients),
            client_samples=client_samples
        )
        end_time = time.time()
        
        results[num_clients] = {
            "total_time": end_time - start_time,
            "avg_round_time": (end_time - start_time) / num_rounds,
            "final_loss": history.losses_centralized[-1][1] if history.losses_centralized else None,
            "final_mae": history.metrics_centralized["mae"][-1][1] if history.metrics_centralized and "mae" in history.metrics_centralized else None
        }
    
    # Print benchmark results
    print("\n" + "="*80)
    print("FEDERATED LEARNING PERFORMANCE BENCHMARK")
    print("="*80)
    print(f"{'Clients':<8} {'Total Time':<12} {'Avg Round':<12} {'Final Loss':<12} {'Final MAE':<12}")
    print("-"*80)
    
    for num_clients, metrics in results.items():
        print(f"{num_clients:<8} {metrics['total_time']:<12.2f} {metrics['avg_round_time']:<12.2f} "
              f"{metrics['final_loss']:<12.4f} {metrics['final_mae']:<12.4f}")
    
    print("="*80)
    
    return results


def main():
    """Main function for simulation script."""
    
    parser = argparse.ArgumentParser(description="Federated Learning Simulation")
    parser.add_argument("--clients", type=int, default=3, help="Number of clients")
    parser.add_argument("--rounds", type=int, default=5, help="Number of rounds")
    parser.add_argument("--samples", type=int, default=800, help="Samples per client")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    parser.add_argument("--min-fit", type=int, default=2, help="Minimum clients for training")
    parser.add_argument("--min-eval", type=int, default=2, help="Minimum clients for evaluation")
    
    args = parser.parse_args()
    
    # Configure logging
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
    
    if args.benchmark:
        benchmark_performance(
            num_clients_list=[2, 3, 5],
            num_rounds=args.rounds,
            client_samples=args.samples
        )
    else:
        run_simulation(
            num_clients=args.clients,
            num_rounds=args.rounds,
            min_fit_clients=args.min_fit,
            min_evaluate_clients=args.min_eval,
            min_available_clients=args.min_fit,
            client_samples=args.samples
        )


if __name__ == "__main__":
    main()

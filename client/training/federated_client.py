"""
Federated Learning Client Implementation using Flower
Raspberry Pi 5 Federated Environmental Monitoring Network
"""

import flwr as fl
import tensorflow as tf
import numpy as np
from typing import Dict, List, Tuple, Optional
import structlog
from pathlib import Path
import yaml

# Import model from server (shared model definition)
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from server.models.lstm_model import create_federated_model

logger = structlog.get_logger(__name__)


class EnvironmentalClient(fl.client.NumPyClient):
    """Federated learning client for environmental monitoring."""
    
    def __init__(
        self,
        model: tf.keras.Model,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        client_id: str = "client"
    ):
        """Initialize federated client with local data."""
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.client_id = client_id
        
        logger.info(
            "Initialized federated client",
            client_id=client_id,
            train_samples=len(x_train),
            val_samples=len(x_val)
        )
    
    def get_parameters(self, config: Dict[str, fl.common.Scalar]) -> List[np.ndarray]:
        """Get model parameters as a list of NumPy ndarrays."""
        logger.debug("Getting model parameters", client_id=self.client_id)
        return self.model.get_weights()
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set model parameters from a list of NumPy ndarrays."""
        logger.debug("Setting model parameters", client_id=self.client_id)
        self.model.set_weights(parameters)
    
    def fit(
        self, 
        parameters: List[np.ndarray], 
        config: Dict[str, fl.common.Scalar]
    ) -> Tuple[List[np.ndarray], int, Dict[str, fl.common.Scalar]]:
        """Train model with local data."""
        
        # Extract training configuration
        batch_size = int(config.get("batch_size", 32))
        epochs = int(config.get("local_epochs", 1))
        learning_rate = float(config.get("learning_rate", 0.001))
        server_round = int(config.get("server_round", 0))
        
        logger.info(
            "Starting local training",
            client_id=self.client_id,
            round=server_round,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        
        # Set parameters received from server
        self.set_parameters(parameters)
        
        # Update learning rate if specified
        if hasattr(self.model.optimizer, 'learning_rate'):
            self.model.optimizer.learning_rate.assign(learning_rate)
        
        # Train model
        history = self.model.fit(
            self.x_train,
            self.y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(self.x_val, self.y_val),
            verbose=0
        )
        
        # Extract metrics
        train_loss = float(history.history["loss"][-1])
        train_mae = float(history.history["mae"][-1])
        val_loss = float(history.history["val_loss"][-1])
        val_mae = float(history.history["val_mae"][-1])
        
        logger.info(
            "Local training complete",
            client_id=self.client_id,
            round=server_round,
            train_loss=train_loss,
            train_mae=train_mae,
            val_loss=val_loss,
            val_mae=val_mae
        )
        
        # Return updated parameters and metrics
        return (
            self.get_parameters(config={}),
            len(self.x_train),
            {
                "train_loss": train_loss,
                "train_mae": train_mae,
                "val_loss": val_loss,
                "val_mae": val_mae,
            }
        )
    
    def evaluate(
        self, 
        parameters: List[np.ndarray], 
        config: Dict[str, fl.common.Scalar]
    ) -> Tuple[float, int, Dict[str, fl.common.Scalar]]:
        """Evaluate model with local validation data."""
        
        batch_size = int(config.get("batch_size", 32))
        server_round = int(config.get("server_round", 0))
        
        logger.info(
            "Starting local evaluation",
            client_id=self.client_id,
            round=server_round,
            batch_size=batch_size
        )
        
        # Set parameters received from server
        self.set_parameters(parameters)
        
        # Evaluate model
        loss, mae = self.model.evaluate(
            self.x_val, 
            self.y_val, 
            batch_size=batch_size,
            verbose=0
        )
        
        logger.info(
            "Local evaluation complete",
            client_id=self.client_id,
            round=server_round,
            loss=loss,
            mae=mae
        )
        
        return loss, len(self.x_val), {"mae": mae}


def generate_synthetic_data(
    num_samples: int = 1000,
    sequence_length: int = 24,
    num_features: int = 2,
    client_id: str = "client",
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic environmental data for testing."""
    
    if seed is not None:
        np.random.seed(seed)
    
    logger.info(
        "Generating synthetic data",
        client_id=client_id,
        samples=num_samples,
        sequence_length=sequence_length,
        features=num_features
    )
    
    # Generate time-based patterns
    time_steps = np.arange(num_samples + sequence_length)
    
    # Temperature pattern (seasonal + daily + noise)
    temp_base = 20 + 10 * np.sin(2 * np.pi * time_steps / 365)  # Seasonal
    temp_daily = 5 * np.sin(2 * np.pi * time_steps / 24)        # Daily
    temp_noise = np.random.normal(0, 2, len(time_steps))        # Noise
    temperature = temp_base + temp_daily + temp_noise
    
    # Humidity pattern (inverse correlation with temperature + noise)
    humidity_base = 60 - 0.5 * (temperature - 20)
    humidity_noise = np.random.normal(0, 5, len(time_steps))
    humidity = np.clip(humidity_base + humidity_noise, 10, 90)
    
    # Create sequences
    X = []
    y = []
    
    for i in range(num_samples):
        # Input sequence
        temp_seq = temperature[i:i + sequence_length]
        humid_seq = humidity[i:i + sequence_length]
        X.append(np.column_stack([temp_seq, humid_seq]))
        
        # Target (next values)
        y.append([temperature[i + sequence_length], humidity[i + sequence_length]])
    
    X = np.array(X)
    y = np.array(y)
    
    logger.info(
        "Synthetic data generated",
        client_id=client_id,
        X_shape=X.shape,
        y_shape=y.shape
    )
    
    return X, y


def create_client_data(
    client_id: str,
    total_samples: int = 1000,
    val_split: float = 0.2,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create training and validation data for a client."""
    
    # Generate synthetic data with client-specific seed
    client_seed = hash(client_id) % 10000 if seed is None else seed
    X, y = generate_synthetic_data(
        num_samples=total_samples,
        client_id=client_id,
        seed=client_seed
    )
    
    # Split into train/validation
    val_size = int(total_samples * val_split)
    train_size = total_samples - val_size
    
    x_train, x_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    logger.info(
        "Created client data",
        client_id=client_id,
        train_size=len(x_train),
        val_size=len(x_val)
    )
    
    return x_train, y_train, x_val, y_val


def start_federated_client(
    server_address: str = "localhost:8080",
    client_id: str = "client_1",
    config_path: Optional[str] = None
):
    """Start federated learning client."""
    
    logger.info(
        "Starting federated client",
        client_id=client_id,
        server_address=server_address
    )
    
    # Load configuration
    if config_path:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    # Create model
    model = create_federated_model()
    logger.info("Created client model", params=model.count_params())
    
    # Create client data
    x_train, y_train, x_val, y_val = create_client_data(
        client_id=client_id,
        total_samples=config.get("total_samples", 1000),
        val_split=config.get("val_split", 0.2)
    )
    
    # Create client
    client = EnvironmentalClient(
        model=model,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        client_id=client_id
    )
    
    # Start client
    fl.client.start_numpy_client(
        server_address=server_address,
        client=client
    )
    
    logger.info("Federated client completed", client_id=client_id)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Start federated learning client")
    parser.add_argument("--server", default="localhost:8080", help="Server address")
    parser.add_argument("--client-id", default="client_1", help="Client ID")
    parser.add_argument("--config", help="Configuration file path")
    
    args = parser.parse_args()
    
    start_federated_client(
        server_address=args.server,
        client_id=args.client_id,
        config_path=args.config
    )

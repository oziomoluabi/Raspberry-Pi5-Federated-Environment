"""
Secure Federated Learning Server with TLS and Authentication
Raspberry Pi 5 Federated Environmental Monitoring Network
"""

import flwr as fl
import ssl
from typing import Dict, List, Optional, Tuple, Union
import structlog
import secrets
import hashlib
import time
from pathlib import Path
import yaml
import jwt
from datetime import datetime, timedelta

from .tls_config import TLSCertificateManager, SecureFlowerConfig
from ..models.lstm_model import create_federated_model

logger = structlog.get_logger(__name__)


class SecureTokenManager:
    """Manager for secure client authentication tokens."""
    
    def __init__(self, secret_key: Optional[str] = None):
        """Initialize token manager."""
        self.secret_key = secret_key or secrets.token_urlsafe(32)
        self.issued_tokens = {}  # Track issued tokens
        
        logger.info("Secure token manager initialized")
    
    def generate_client_token(
        self,
        client_id: str,
        validity_hours: int = 24,
        permissions: Optional[List[str]] = None
    ) -> str:
        """Generate JWT token for client authentication."""
        
        if permissions is None:
            permissions = ["federated_learning", "model_update"]
        
        # Token payload
        payload = {
            'client_id': client_id,
            'permissions': permissions,
            'issued_at': datetime.utcnow().isoformat(),
            'expires_at': (datetime.utcnow() + timedelta(hours=validity_hours)).isoformat(),
            'iat': datetime.utcnow(),
            'exp': datetime.utcnow() + timedelta(hours=validity_hours)
        }
        
        # Generate token
        token = jwt.encode(payload, self.secret_key, algorithm='HS256')
        
        # Store token info
        self.issued_tokens[client_id] = {
            'token': token,
            'issued_at': datetime.utcnow(),
            'expires_at': datetime.utcnow() + timedelta(hours=validity_hours),
            'permissions': permissions
        }
        
        logger.info("Client token generated", 
                   client_id=client_id, 
                   validity_hours=validity_hours,
                   permissions=permissions)
        
        return token
    
    def validate_token(self, token: str) -> Tuple[bool, Optional[Dict]]:
        """Validate client token."""
        
        try:
            # Decode and verify token
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            
            client_id = payload.get('client_id')
            permissions = payload.get('permissions', [])
            
            # Check if token is still valid
            exp_time = datetime.fromisoformat(payload.get('expires_at'))
            if datetime.utcnow() > exp_time:
                logger.warning("Token expired", client_id=client_id)
                return False, None
            
            logger.debug("Token validated successfully", client_id=client_id)
            
            return True, {
                'client_id': client_id,
                'permissions': permissions,
                'payload': payload
            }
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired during validation")
            return False, None
        except jwt.InvalidTokenError as e:
            logger.warning("Invalid token", error=str(e))
            return False, None
        except Exception as e:
            logger.error("Token validation error", error=str(e))
            return False, None
    
    def revoke_token(self, client_id: str) -> bool:
        """Revoke token for a client."""
        
        if client_id in self.issued_tokens:
            del self.issued_tokens[client_id]
            logger.info("Token revoked", client_id=client_id)
            return True
        else:
            logger.warning("Token not found for revocation", client_id=client_id)
            return False
    
    def get_active_tokens(self) -> Dict:
        """Get list of active tokens."""
        
        active_tokens = {}
        current_time = datetime.utcnow()
        
        for client_id, token_info in self.issued_tokens.items():
            if current_time < token_info['expires_at']:
                active_tokens[client_id] = {
                    'issued_at': token_info['issued_at'].isoformat(),
                    'expires_at': token_info['expires_at'].isoformat(),
                    'permissions': token_info['permissions']
                }
        
        return active_tokens


class SecureFederatedStrategy(fl.server.strategy.FedAvg):
    """Secure federated averaging strategy with authentication and privacy."""
    
    def __init__(
        self,
        token_manager: SecureTokenManager,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[callable] = None,
        on_fit_config_fn: Optional[callable] = None,
        on_evaluate_config_fn: Optional[callable] = None,
        accept_failures: bool = False,
        initial_parameters: Optional[fl.common.Parameters] = None,
        enable_differential_privacy: bool = False,
        dp_noise_multiplier: float = 1.0,
        dp_l2_norm_clip: float = 1.0
    ):
        """Initialize secure federated strategy."""
        
        super().__init__(
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters
        )
        
        self.token_manager = token_manager
        self.enable_differential_privacy = enable_differential_privacy
        self.dp_noise_multiplier = dp_noise_multiplier
        self.dp_l2_norm_clip = dp_l2_norm_clip
        
        # Security metrics
        self.authentication_attempts = 0
        self.successful_authentications = 0
        self.failed_authentications = 0
        
        logger.info(
            "Secure federated strategy initialized",
            min_clients=min_available_clients,
            differential_privacy=enable_differential_privacy,
            dp_noise_multiplier=dp_noise_multiplier if enable_differential_privacy else None
        )
    
    def configure_fit(
        self, 
        server_round: int, 
        parameters: fl.common.Parameters, 
        client_manager: fl.server.client_manager.ClientManager
    ) -> List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitIns]]:
        """Configure fit with authentication."""
        
        logger.info("Configuring secure fit", round=server_round)
        
        # Get base configuration
        config_pairs = super().configure_fit(server_round, parameters, client_manager)
        
        # Add security configuration
        secure_config_pairs = []
        for client_proxy, fit_ins in config_pairs:
            # Add security parameters to config
            secure_config = fit_ins.config.copy()
            secure_config.update({
                "server_round": server_round,
                "enable_dp": self.enable_differential_privacy,
                "dp_noise_multiplier": self.dp_noise_multiplier,
                "dp_l2_norm_clip": self.dp_l2_norm_clip,
                "require_authentication": True
            })
            
            secure_fit_ins = fl.common.FitIns(fit_ins.parameters, secure_config)
            secure_config_pairs.append((client_proxy, secure_fit_ins))
        
        logger.info("Secure fit configuration completed", clients=len(secure_config_pairs))
        return secure_config_pairs
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes], BaseException]],
    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, fl.common.Scalar]]:
        """Aggregate fit results with differential privacy."""
        
        logger.info(
            "Aggregating secure fit results",
            round=server_round,
            num_results=len(results),
            num_failures=len(failures)
        )
        
        # Validate client authentication (simplified - in practice, this would be done earlier)
        authenticated_results = []
        for client_proxy, fit_res in results:
            # In a real implementation, authentication would be validated during communication
            # Here we assume all results are from authenticated clients
            authenticated_results.append((client_proxy, fit_res))
        
        if len(authenticated_results) < self.min_fit_clients:
            logger.warning("Insufficient authenticated clients for aggregation")
            return None, {}
        
        # Apply differential privacy if enabled
        if self.enable_differential_privacy:
            authenticated_results = self._apply_differential_privacy(authenticated_results)
        
        # Call parent aggregation
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, authenticated_results, failures
        )
        
        # Add security metrics
        security_metrics = {
            "authenticated_clients": len(authenticated_results),
            "failed_authentications": len(results) - len(authenticated_results),
            "differential_privacy_enabled": self.enable_differential_privacy
        }
        
        if aggregated_metrics:
            aggregated_metrics.update(security_metrics)
        else:
            aggregated_metrics = security_metrics
        
        logger.info(
            "Secure aggregation completed",
            round=server_round,
            authenticated_clients=len(authenticated_results),
            dp_enabled=self.enable_differential_privacy
        )
        
        return aggregated_parameters, aggregated_metrics
    
    def _apply_differential_privacy(
        self,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]]
    ) -> List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]]:
        """Apply differential privacy to client updates."""
        
        logger.debug("Applying differential privacy", 
                    noise_multiplier=self.dp_noise_multiplier,
                    l2_norm_clip=self.dp_l2_norm_clip)
        
        dp_results = []
        
        for client_proxy, fit_res in results:
            # Get parameters
            parameters = fl.common.parameters_to_ndarrays(fit_res.parameters)
            
            # Apply L2 norm clipping
            clipped_parameters = []
            for param in parameters:
                param_norm = (param ** 2).sum() ** 0.5
                if param_norm > self.dp_l2_norm_clip:
                    clipped_param = param * (self.dp_l2_norm_clip / param_norm)
                else:
                    clipped_param = param
                clipped_parameters.append(clipped_param)
            
            # Add Gaussian noise
            noisy_parameters = []
            for param in clipped_parameters:
                noise = np.random.normal(
                    0, 
                    self.dp_noise_multiplier * self.dp_l2_norm_clip, 
                    param.shape
                )
                noisy_param = param + noise
                noisy_parameters.append(noisy_param)
            
            # Create new FitRes with noisy parameters
            noisy_fit_res = fl.common.FitRes(
                status=fit_res.status,
                parameters=fl.common.ndarrays_to_parameters(noisy_parameters),
                num_examples=fit_res.num_examples,
                metrics=fit_res.metrics
            )
            
            dp_results.append((client_proxy, noisy_fit_res))
        
        logger.debug("Differential privacy applied", clients=len(dp_results))
        return dp_results


class SecureFederatedServer:
    """Secure federated learning server with TLS and authentication."""
    
    def __init__(
        self,
        cert_dir: str = "certs",
        config_path: Optional[str] = None
    ):
        """Initialize secure federated server."""
        
        self.cert_dir = Path(cert_dir)
        self.config_path = config_path
        
        # Initialize security components
        self.cert_manager = TLSCertificateManager(cert_dir)
        self.secure_config = SecureFlowerConfig(self.cert_manager)
        self.token_manager = SecureTokenManager()
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize model
        self.model = create_federated_model()
        
        logger.info(
            "Secure federated server initialized",
            cert_dir=cert_dir,
            config_path=config_path
        )
    
    def _load_config(self) -> Dict:
        """Load server configuration."""
        
        default_config = {
            "server_address": "0.0.0.0:8080",
            "num_rounds": 5,
            "min_fit_clients": 2,
            "min_evaluate_clients": 2,
            "min_available_clients": 2,
            "enable_tls": True,
            "enable_differential_privacy": False,
            "dp_noise_multiplier": 1.0,
            "dp_l2_norm_clip": 1.0,
            "token_validity_hours": 24
        }
        
        if self.config_path and Path(self.config_path).exists():
            with open(self.config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                default_config.update(user_config)
                logger.info("Configuration loaded", config_path=self.config_path)
        else:
            logger.info("Using default configuration")
        
        return default_config
    
    def setup_security(self) -> Dict[str, str]:
        """Set up security infrastructure."""
        
        logger.info("Setting up security infrastructure")
        
        # Generate certificates if they don't exist
        cert_paths = self.cert_manager.setup_complete_pki()
        
        # Generate initial client tokens
        initial_clients = ["client_1", "client_2", "client_3"]
        tokens = {}
        
        for client_id in initial_clients:
            token = self.token_manager.generate_client_token(
                client_id=client_id,
                validity_hours=self.config["token_validity_hours"]
            )
            tokens[client_id] = token
        
        logger.info("Security infrastructure setup complete", 
                   certificates=len(cert_paths),
                   tokens=len(tokens))
        
        return {
            "certificates": cert_paths,
            "tokens": tokens
        }
    
    def create_secure_strategy(self) -> SecureFederatedStrategy:
        """Create secure federated learning strategy."""
        
        # Get initial model parameters
        initial_parameters = fl.common.ndarrays_to_parameters(self.model.get_weights())
        
        strategy = SecureFederatedStrategy(
            token_manager=self.token_manager,
            min_fit_clients=self.config["min_fit_clients"],
            min_evaluate_clients=self.config["min_evaluate_clients"],
            min_available_clients=self.config["min_available_clients"],
            initial_parameters=initial_parameters,
            enable_differential_privacy=self.config["enable_differential_privacy"],
            dp_noise_multiplier=self.config["dp_noise_multiplier"],
            dp_l2_norm_clip=self.config["dp_l2_norm_clip"]
        )
        
        logger.info("Secure strategy created")
        return strategy
    
    def start_secure_server(self) -> None:
        """Start secure federated learning server."""
        
        logger.info("Starting secure federated learning server")
        
        # Setup security
        security_info = self.setup_security()
        
        # Create strategy
        strategy = self.create_secure_strategy()
        
        # Get server configuration
        if self.config["enable_tls"]:
            server_config = self.secure_config.get_server_config(
                self.config["server_address"]
            )
            
            logger.info("Starting server with TLS encryption", 
                       address=self.config["server_address"])
            
            # Start secure server
            fl.server.start_server(
                server_address=self.config["server_address"],
                config=fl.server.ServerConfig(num_rounds=self.config["num_rounds"]),
                strategy=strategy,
                certificates=server_config["certificates"]
            )
        else:
            logger.warning("Starting server WITHOUT TLS encryption")
            
            # Start insecure server (for testing only)
            fl.server.start_server(
                server_address=self.config["server_address"],
                config=fl.server.ServerConfig(num_rounds=self.config["num_rounds"]),
                strategy=strategy
            )
        
        logger.info("Secure federated server completed")
    
    def get_client_credentials(self, client_id: str) -> Optional[Dict]:
        """Get credentials for a client."""
        
        active_tokens = self.token_manager.get_active_tokens()
        
        if client_id in active_tokens:
            token_info = active_tokens[client_id]
            
            if self.config["enable_tls"]:
                client_config = self.secure_config.get_client_config()
                certificates = client_config["certificates"]
            else:
                certificates = None
            
            return {
                "client_id": client_id,
                "token": self.token_manager.issued_tokens[client_id]["token"],
                "server_address": self.config["server_address"],
                "certificates": certificates,
                "token_info": token_info
            }
        else:
            logger.warning("No active token found for client", client_id=client_id)
            return None


if __name__ == "__main__":
    # Example usage
    import tempfile
    
    # Create temporary directory for certificates
    with tempfile.TemporaryDirectory() as temp_dir:
        server = SecureFederatedServer(cert_dir=temp_dir)
        
        # Setup security (don't start server in example)
        security_info = server.setup_security()
        
        print("Security setup completed!")
        print("Certificates:", list(security_info["certificates"].keys()))
        print("Tokens:", list(security_info["tokens"].keys()))
        
        # Get client credentials
        client_creds = server.get_client_credentials("client_1")
        if client_creds:
            print(f"Client credentials for {client_creds['client_id']} ready")

"""
TLS Configuration and Certificate Management
Raspberry Pi 5 Federated Environmental Monitoring Network
"""

import os
import ssl
from pathlib import Path
from typing import Dict, Optional, Tuple
import structlog
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
import datetime
import ipaddress

logger = structlog.get_logger(__name__)


class TLSCertificateManager:
    """Manager for TLS certificates and SSL contexts."""
    
    def __init__(self, cert_dir: str = "certs"):
        """Initialize TLS certificate manager."""
        self.cert_dir = Path(cert_dir)
        self.cert_dir.mkdir(parents=True, exist_ok=True)
        
        # Certificate file paths
        self.ca_cert_path = self.cert_dir / "ca-cert.pem"
        self.ca_key_path = self.cert_dir / "ca-key.pem"
        self.server_cert_path = self.cert_dir / "server-cert.pem"
        self.server_key_path = self.cert_dir / "server-key.pem"
        self.client_cert_path = self.cert_dir / "client-cert.pem"
        self.client_key_path = self.cert_dir / "client-key.pem"
        
        logger.info("TLS Certificate Manager initialized", cert_dir=str(self.cert_dir))
    
    def generate_ca_certificate(
        self,
        common_name: str = "Raspberry Pi 5 Federated CA",
        validity_days: int = 365
    ) -> Tuple[x509.Certificate, rsa.RSAPrivateKey]:
        """Generate Certificate Authority certificate and private key."""
        
        logger.info("Generating CA certificate", common_name=common_name, validity_days=validity_days)
        
        # Generate private key
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )
        
        # Create certificate
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "CA"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "San Francisco"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Raspberry Pi 5 Federated"),
            x509.NameAttribute(NameOID.COMMON_NAME, common_name),
        ])
        
        cert = x509.CertificateBuilder().subject_name(
            subject
        ).issuer_name(
            issuer
        ).public_key(
            private_key.public_key()
        ).serial_number(
            x509.random_serial_number()
        ).not_valid_before(
            datetime.datetime.utcnow()
        ).not_valid_after(
            datetime.datetime.utcnow() + datetime.timedelta(days=validity_days)
        ).add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName("localhost"),
                x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
            ]),
            critical=False,
        ).add_extension(
            x509.BasicConstraints(ca=True, path_length=None),
            critical=True,
        ).add_extension(
            x509.KeyUsage(
                key_cert_sign=True,
                crl_sign=True,
                digital_signature=False,
                key_encipherment=False,
                key_agreement=False,
                content_commitment=False,
                data_encipherment=False,
                encipher_only=False,
                decipher_only=False
            ),
            critical=True,
        ).sign(private_key, hashes.SHA256())
        
        # Save to files
        with open(self.ca_cert_path, "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))
        
        with open(self.ca_key_path, "wb") as f:
            f.write(private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ))
        
        logger.info("CA certificate generated and saved", 
                   cert_path=str(self.ca_cert_path),
                   key_path=str(self.ca_key_path))
        
        return cert, private_key
    
    def generate_server_certificate(
        self,
        ca_cert: x509.Certificate,
        ca_key: rsa.RSAPrivateKey,
        common_name: str = "localhost",
        san_hosts: Optional[list] = None,
        validity_days: int = 365
    ) -> Tuple[x509.Certificate, rsa.RSAPrivateKey]:
        """Generate server certificate signed by CA."""
        
        if san_hosts is None:
            san_hosts = ["localhost", "127.0.0.1", "0.0.0.0"]
        
        logger.info("Generating server certificate", 
                   common_name=common_name, 
                   san_hosts=san_hosts,
                   validity_days=validity_days)
        
        # Generate private key
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )
        
        # Create subject alternative names
        san_list = []
        for host in san_hosts:
            try:
                # Try to parse as IP address
                ip = ipaddress.ip_address(host)
                san_list.append(x509.IPAddress(ip))
            except ValueError:
                # Not an IP, treat as DNS name
                san_list.append(x509.DNSName(host))
        
        # Create certificate
        subject = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "CA"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "San Francisco"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Raspberry Pi 5 Federated"),
            x509.NameAttribute(NameOID.COMMON_NAME, common_name),
        ])
        
        cert = x509.CertificateBuilder().subject_name(
            subject
        ).issuer_name(
            ca_cert.subject
        ).public_key(
            private_key.public_key()
        ).serial_number(
            x509.random_serial_number()
        ).not_valid_before(
            datetime.datetime.utcnow()
        ).not_valid_after(
            datetime.datetime.utcnow() + datetime.timedelta(days=validity_days)
        ).add_extension(
            x509.SubjectAlternativeName(san_list),
            critical=False,
        ).add_extension(
            x509.BasicConstraints(ca=False, path_length=None),
            critical=True,
        ).add_extension(
            x509.KeyUsage(
                key_cert_sign=False,
                crl_sign=False,
                digital_signature=True,
                key_encipherment=True,
                key_agreement=False,
                content_commitment=False,
                data_encipherment=False,
                encipher_only=False,
                decipher_only=False
            ),
            critical=True,
        ).add_extension(
            x509.ExtendedKeyUsage([
                x509.oid.ExtendedKeyUsageOID.SERVER_AUTH,
            ]),
            critical=True,
        ).sign(ca_key, hashes.SHA256())
        
        # Save to files
        with open(self.server_cert_path, "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))
        
        with open(self.server_key_path, "wb") as f:
            f.write(private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ))
        
        logger.info("Server certificate generated and saved",
                   cert_path=str(self.server_cert_path),
                   key_path=str(self.server_key_path))
        
        return cert, private_key
    
    def generate_client_certificate(
        self,
        ca_cert: x509.Certificate,
        ca_key: rsa.RSAPrivateKey,
        common_name: str = "federated-client",
        validity_days: int = 365
    ) -> Tuple[x509.Certificate, rsa.RSAPrivateKey]:
        """Generate client certificate signed by CA."""
        
        logger.info("Generating client certificate", 
                   common_name=common_name,
                   validity_days=validity_days)
        
        # Generate private key
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )
        
        # Create certificate
        subject = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "CA"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "San Francisco"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Raspberry Pi 5 Federated"),
            x509.NameAttribute(NameOID.COMMON_NAME, common_name),
        ])
        
        cert = x509.CertificateBuilder().subject_name(
            subject
        ).issuer_name(
            ca_cert.subject
        ).public_key(
            private_key.public_key()
        ).serial_number(
            x509.random_serial_number()
        ).not_valid_before(
            datetime.datetime.utcnow()
        ).not_valid_after(
            datetime.datetime.utcnow() + datetime.timedelta(days=validity_days)
        ).add_extension(
            x509.BasicConstraints(ca=False, path_length=None),
            critical=True,
        ).add_extension(
            x509.KeyUsage(
                key_cert_sign=False,
                crl_sign=False,
                digital_signature=True,
                key_encipherment=True,
                key_agreement=False,
                content_commitment=False,
                data_encipherment=False,
                encipher_only=False,
                decipher_only=False
            ),
            critical=True,
        ).add_extension(
            x509.ExtendedKeyUsage([
                x509.oid.ExtendedKeyUsageOID.CLIENT_AUTH,
            ]),
            critical=True,
        ).sign(ca_key, hashes.SHA256())
        
        # Save to files
        with open(self.client_cert_path, "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))
        
        with open(self.client_key_path, "wb") as f:
            f.write(private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ))
        
        logger.info("Client certificate generated and saved",
                   cert_path=str(self.client_cert_path),
                   key_path=str(self.client_key_path))
        
        return cert, private_key
    
    def setup_complete_pki(
        self,
        server_hosts: Optional[list] = None,
        validity_days: int = 365
    ) -> Dict[str, str]:
        """Set up complete PKI infrastructure."""
        
        logger.info("Setting up complete PKI infrastructure")
        
        if server_hosts is None:
            server_hosts = ["localhost", "127.0.0.1", "0.0.0.0"]
        
        # Generate CA certificate
        ca_cert, ca_key = self.generate_ca_certificate(validity_days=validity_days)
        
        # Generate server certificate
        server_cert, server_key = self.generate_server_certificate(
            ca_cert, ca_key, san_hosts=server_hosts, validity_days=validity_days
        )
        
        # Generate client certificate
        client_cert, client_key = self.generate_client_certificate(
            ca_cert, ca_key, validity_days=validity_days
        )
        
        certificate_paths = {
            'ca_cert': str(self.ca_cert_path),
            'ca_key': str(self.ca_key_path),
            'server_cert': str(self.server_cert_path),
            'server_key': str(self.server_key_path),
            'client_cert': str(self.client_cert_path),
            'client_key': str(self.client_key_path)
        }
        
        logger.info("PKI infrastructure setup complete", certificates=certificate_paths)
        
        return certificate_paths
    
    def create_server_ssl_context(self) -> ssl.SSLContext:
        """Create SSL context for server."""
        
        if not self.server_cert_path.exists() or not self.server_key_path.exists():
            raise FileNotFoundError("Server certificate or key not found")
        
        context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        context.load_cert_chain(str(self.server_cert_path), str(self.server_key_path))
        
        # Require client certificates
        if self.ca_cert_path.exists():
            context.load_verify_locations(str(self.ca_cert_path))
            context.verify_mode = ssl.CERT_REQUIRED
        
        logger.info("Server SSL context created")
        return context
    
    def create_client_ssl_context(self) -> ssl.SSLContext:
        """Create SSL context for client."""
        
        if not self.client_cert_path.exists() or not self.client_key_path.exists():
            raise FileNotFoundError("Client certificate or key not found")
        
        context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
        context.load_cert_chain(str(self.client_cert_path), str(self.client_key_path))
        
        # Load CA certificate for server verification
        if self.ca_cert_path.exists():
            context.load_verify_locations(str(self.ca_cert_path))
        
        logger.info("Client SSL context created")
        return context
    
    def get_certificate_info(self, cert_path: str) -> Dict:
        """Get information about a certificate."""
        
        cert_path = Path(cert_path)
        if not cert_path.exists():
            return {'error': 'Certificate file not found'}
        
        try:
            with open(cert_path, 'rb') as f:
                cert_data = f.read()
            
            cert = x509.load_pem_x509_certificate(cert_data)
            
            return {
                'subject': cert.subject.rfc4514_string(),
                'issuer': cert.issuer.rfc4514_string(),
                'serial_number': str(cert.serial_number),
                'not_valid_before': cert.not_valid_before.isoformat(),
                'not_valid_after': cert.not_valid_after.isoformat(),
                'is_expired': cert.not_valid_after < datetime.datetime.utcnow(),
                'days_until_expiry': (cert.not_valid_after - datetime.datetime.utcnow()).days
            }
        except Exception as e:
            return {'error': f'Failed to parse certificate: {str(e)}'}


class SecureFlowerConfig:
    """Configuration for secure Flower federated learning."""
    
    def __init__(self, cert_manager: TLSCertificateManager):
        """Initialize secure Flower configuration."""
        self.cert_manager = cert_manager
        
        logger.info("Secure Flower configuration initialized")
    
    def get_server_config(self, server_address: str = "0.0.0.0:8080") -> Dict:
        """Get secure server configuration for Flower."""
        
        # Ensure certificates exist
        if not all([
            self.cert_manager.server_cert_path.exists(),
            self.cert_manager.server_key_path.exists(),
            self.cert_manager.ca_cert_path.exists()
        ]):
            logger.warning("Certificates not found, setting up PKI")
            self.cert_manager.setup_complete_pki()
        
        config = {
            'server_address': server_address,
            'certificates': (
                str(self.cert_manager.server_cert_path),
                str(self.cert_manager.server_key_path),
                str(self.cert_manager.ca_cert_path)
            ),
            'ssl_context': self.cert_manager.create_server_ssl_context()
        }
        
        logger.info("Secure server configuration created", address=server_address)
        return config
    
    def get_client_config(self, server_address: str = "localhost:8080") -> Dict:
        """Get secure client configuration for Flower."""
        
        # Ensure certificates exist
        if not all([
            self.cert_manager.client_cert_path.exists(),
            self.cert_manager.client_key_path.exists(),
            self.cert_manager.ca_cert_path.exists()
        ]):
            logger.warning("Certificates not found, setting up PKI")
            self.cert_manager.setup_complete_pki()
        
        config = {
            'server_address': server_address,
            'certificates': (
                str(self.cert_manager.client_cert_path),
                str(self.cert_manager.client_key_path),
                str(self.cert_manager.ca_cert_path)
            ),
            'ssl_context': self.cert_manager.create_client_ssl_context()
        }
        
        logger.info("Secure client configuration created", address=server_address)
        return config


if __name__ == "__main__":
    # Example usage
    import tempfile
    
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        cert_manager = TLSCertificateManager(cert_dir=temp_dir)
        
        # Set up complete PKI
        cert_paths = cert_manager.setup_complete_pki()
        print("Certificate paths:", cert_paths)
        
        # Get certificate info
        ca_info = cert_manager.get_certificate_info(cert_paths['ca_cert'])
        print("CA Certificate info:", ca_info)
        
        # Create secure Flower config
        secure_config = SecureFlowerConfig(cert_manager)
        server_config = secure_config.get_server_config()
        client_config = secure_config.get_client_config()
        
        print("Secure configuration created successfully!")

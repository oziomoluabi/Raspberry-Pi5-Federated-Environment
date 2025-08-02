# Security Guide

This guide covers security best practices, configuration, and deployment guidelines for the Raspberry Pi 5 Federated Environmental Monitoring Network.

## Table of Contents

1. [Security Overview](#security-overview)
2. [TLS Configuration](#tls-configuration)
3. [Authentication & Authorization](#authentication--authorization)
4. [Differential Privacy](#differential-privacy)
5. [Secure Deployment](#secure-deployment)
6. [Monitoring & Auditing](#monitoring--auditing)
7. [Incident Response](#incident-response)

## Security Overview

The federated learning system implements multiple layers of security:

- **Transport Security**: TLS 1.3 encryption for all communications
- **Authentication**: JWT-based client authentication
- **Privacy Protection**: Differential privacy for model updates
- **Code Security**: Automated vulnerability scanning and secure coding practices
- **Infrastructure Security**: Certificate management and secure deployment

## TLS Configuration

### Certificate Management

The system uses a complete PKI infrastructure:

```python
from server.security.tls_config import TLSCertificateManager

# Initialize certificate manager
cert_manager = TLSCertificateManager(cert_dir="certs")

# Setup complete PKI (CA, server, and client certificates)
cert_paths = cert_manager.setup_complete_pki(
    server_hosts=["localhost", "192.168.1.100", "federated-server.local"],
    validity_days=365
)
```

### Server Configuration

Enable TLS on the federated learning server:

```python
from server.security.secure_federated_server import SecureFederatedServer

# Create secure server with TLS enabled
server = SecureFederatedServer(cert_dir="certs")

# Configuration in config.yaml
config = {
    "enable_tls": True,
    "server_address": "0.0.0.0:8080",
    "min_fit_clients": 2,
    "token_validity_hours": 24
}
```

### Client Configuration

Configure clients for secure communication:

```python
from server.security.tls_config import SecureFlowerConfig

# Get client credentials
client_creds = server.get_client_credentials("client_1")

# Use credentials for secure connection
import flwr as fl
fl.client.start_numpy_client(
    server_address=client_creds["server_address"],
    client=your_client,
    certificates=client_creds["certificates"]
)
```

## Authentication & Authorization

### JWT Token Management

The system uses JWT tokens for client authentication:

```python
from server.security.secure_federated_server import SecureTokenManager

# Initialize token manager
token_manager = SecureTokenManager()

# Generate client token
token = token_manager.generate_client_token(
    client_id="client_1",
    validity_hours=24,
    permissions=["federated_learning", "model_update"]
)

# Validate token
is_valid, payload = token_manager.validate_token(token)
```

### Permission System

Clients are granted specific permissions:

- `federated_learning`: Participate in federated learning rounds
- `model_update`: Submit model updates
- `evaluation`: Participate in model evaluation
- `admin`: Administrative operations (server-side only)

### Token Lifecycle

1. **Generation**: Tokens are generated with expiration times
2. **Validation**: Each client request validates the token
3. **Renewal**: Tokens can be renewed before expiration
4. **Revocation**: Tokens can be revoked immediately if compromised

## Differential Privacy

### Configuration

Enable differential privacy for enhanced privacy protection:

```python
strategy = SecureFederatedStrategy(
    token_manager=token_manager,
    enable_differential_privacy=True,
    dp_noise_multiplier=1.0,  # Higher values = more privacy, less accuracy
    dp_l2_norm_clip=1.0       # Gradient clipping threshold
)
```

### Privacy Parameters

- **Noise Multiplier**: Controls the amount of noise added (typically 0.5-2.0)
- **L2 Norm Clip**: Clips gradients to prevent large updates (typically 1.0-10.0)
- **Privacy Budget**: Track cumulative privacy loss over time

### Privacy-Accuracy Tradeoff

```python
# Conservative privacy (high privacy, lower accuracy)
dp_config = {
    "dp_noise_multiplier": 2.0,
    "dp_l2_norm_clip": 1.0
}

# Balanced privacy (moderate privacy and accuracy)
dp_config = {
    "dp_noise_multiplier": 1.0,
    "dp_l2_norm_clip": 2.0
}

# Minimal privacy (lower privacy, higher accuracy)
dp_config = {
    "dp_noise_multiplier": 0.5,
    "dp_l2_norm_clip": 5.0
}
```

## Secure Deployment

### Production Checklist

Before deploying to production:

- [ ] Generate production certificates with proper hostnames
- [ ] Configure firewall rules (allow only necessary ports)
- [ ] Set up log monitoring and alerting
- [ ] Enable automatic security updates
- [ ] Configure backup and recovery procedures
- [ ] Test incident response procedures

### Network Security

```bash
# Firewall configuration (example for Ubuntu/Debian)
sudo ufw enable
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 8080/tcp  # Federated learning server
sudo ufw deny 80/tcp     # Block HTTP (use HTTPS only)
```

### Certificate Deployment

```bash
# Secure certificate storage
sudo mkdir -p /etc/federated-learning/certs
sudo chown root:federated /etc/federated-learning/certs
sudo chmod 750 /etc/federated-learning/certs

# Copy certificates with proper permissions
sudo cp certs/*.pem /etc/federated-learning/certs/
sudo chmod 640 /etc/federated-learning/certs/*.pem
```

### Environment Variables

Store sensitive configuration in environment variables:

```bash
# .env file (never commit to version control)
FL_SECRET_KEY=your-secret-key-here
FL_CERT_DIR=/etc/federated-learning/certs
FL_LOG_LEVEL=INFO
FL_ENABLE_TLS=true
FL_ENABLE_DP=true
```

## Monitoring & Auditing

### Security Monitoring

Monitor these security metrics:

- Authentication success/failure rates
- Certificate expiration dates
- Unusual client behavior patterns
- Network connection anomalies
- Resource usage spikes

### Automated Security Scanning

Run regular security scans:

```bash
# Dependency vulnerability scanning
pip-audit --format=json --output=security_report.json

# Code security analysis
bandit -r . -f json -o bandit_report.json

# License compliance check
pip-licenses --format=json --output=licenses.json
```

### Log Analysis

Configure structured logging for security events:

```python
import structlog

logger = structlog.get_logger(__name__)

# Log security events
logger.info("client_authenticated", 
           client_id=client_id, 
           timestamp=time.time(),
           source_ip=request.remote_addr)

logger.warning("authentication_failed",
              client_id=client_id,
              reason="invalid_token",
              timestamp=time.time())
```

## Incident Response

### Security Incident Types

1. **Unauthorized Access**: Invalid authentication attempts
2. **Certificate Compromise**: Stolen or leaked certificates
3. **Data Breach**: Unauthorized access to model data
4. **DoS Attack**: Service disruption attempts
5. **Malicious Client**: Compromised federated learning client

### Response Procedures

#### Immediate Response (0-1 hours)

1. **Assess Impact**: Determine scope and severity
2. **Contain Threat**: Isolate affected systems
3. **Preserve Evidence**: Capture logs and system state
4. **Notify Stakeholders**: Alert relevant team members

#### Short-term Response (1-24 hours)

1. **Revoke Compromised Credentials**: Invalidate tokens/certificates
2. **Apply Patches**: Update vulnerable components
3. **Enhance Monitoring**: Increase logging and alerting
4. **Document Incident**: Record timeline and actions taken

#### Long-term Response (1-7 days)

1. **Root Cause Analysis**: Identify underlying vulnerabilities
2. **Implement Fixes**: Address security gaps
3. **Update Procedures**: Improve security practices
4. **Conduct Training**: Educate team on lessons learned

### Emergency Contacts

Maintain an updated list of emergency contacts:

- Security Team Lead
- System Administrator
- Legal/Compliance Officer
- External Security Consultant (if applicable)

### Recovery Procedures

```bash
# Emergency certificate regeneration
python -c "
from server.security.tls_config import TLSCertificateManager
cert_manager = TLSCertificateManager('emergency_certs')
cert_manager.setup_complete_pki()
print('Emergency certificates generated')
"

# Token revocation for all clients
python -c "
from server.security.secure_federated_server import SecureTokenManager
token_manager = SecureTokenManager()
# Revoke all tokens and regenerate
for client_id in ['client_1', 'client_2', 'client_3']:
    token_manager.revoke_token(client_id)
    new_token = token_manager.generate_client_token(client_id)
    print(f'New token for {client_id}: {new_token[:20]}...')
"
```

## Security Best Practices

### Development

- Use secure coding practices
- Implement input validation
- Follow principle of least privilege
- Regular security code reviews
- Automated security testing in CI/CD

### Deployment

- Use production-grade certificates
- Enable all security features
- Regular security updates
- Monitor security metrics
- Implement backup and recovery

### Operations

- Regular security audits
- Incident response testing
- Security awareness training
- Vendor security assessments
- Compliance monitoring

## Compliance

### Data Protection

- Implement differential privacy
- Minimize data collection
- Secure data transmission
- Regular data audits
- Clear data retention policies

### Regulatory Compliance

Consider relevant regulations:

- GDPR (General Data Protection Regulation)
- CCPA (California Consumer Privacy Act)
- HIPAA (if handling health data)
- Industry-specific regulations

### Audit Trail

Maintain comprehensive audit logs:

- Authentication events
- Authorization decisions
- Data access patterns
- Configuration changes
- Security incidents

---

For questions about security configuration or to report security issues, please refer to our [Security Policy](../SECURITY.md).

*Last updated: August 2, 2025*

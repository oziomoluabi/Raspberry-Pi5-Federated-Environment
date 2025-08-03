"""
Unit Tests for Security Framework
Raspberry Pi 5 Federated Environmental Monitoring Network

Comprehensive tests for the security framework implemented in Sprint 5:
- TLS certificate management and PKI infrastructure
- JWT-based authentication system
- Differential privacy implementation
- Security audit functionality
"""

import pytest
import tempfile
import json
import ssl
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import numpy as np
import jwt

# Import security modules
import sys
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from server.security.tls_config import TLSCertificateManager
from server.security.secure_federated_server import SecureTokenManager
from server.security.secure_federated_server import SecureFederatedStrategy
from scripts.security_audit import SecurityAuditor


class TestTLSCertificateManager:
    """Test TLS certificate management functionality."""
    
    def test_certificate_manager_initialization(self, temp_directory):
        """Test TLS certificate manager initialization."""
        cert_manager = TLSCertificateManager(cert_dir=temp_directory)
        
        assert cert_manager.cert_dir == temp_directory
        assert cert_manager.ca_cert_path == temp_directory / "ca_cert.pem"
        assert cert_manager.ca_key_path == temp_directory / "ca_key.pem"
    
    def test_generate_ca_certificate(self, temp_directory):
        """Test CA certificate generation."""
        cert_manager = TLSCertificateManager(cert_dir=temp_directory)
        
        # Generate CA certificate
        ca_cert, ca_key = cert_manager.generate_ca_certificate()
        
        # Verify certificate files are created
        assert cert_manager.ca_cert_path.exists()
        assert cert_manager.ca_key_path.exists()
        
        # Verify certificate properties
        assert ca_cert.subject.rfc4514_string() == "CN=Federated Learning CA"
        assert ca_cert.issuer.rfc4514_string() == "CN=Federated Learning CA"
        
        # Verify certificate is self-signed
        assert ca_cert.subject == ca_cert.issuer
    
    def test_generate_server_certificate(self, temp_directory):
        """Test server certificate generation."""
        cert_manager = TLSCertificateManager(cert_dir=temp_directory)
        
        # First generate CA certificate
        cert_manager.generate_ca_certificate()
        
        # Generate server certificate
        server_cert, server_key = cert_manager.generate_server_certificate("localhost")
        
        # Verify certificate files are created
        server_cert_path = temp_directory / "server_cert.pem"
        server_key_path = temp_directory / "server_key.pem"
        assert server_cert_path.exists()
        assert server_key_path.exists()
        
        # Verify certificate properties
        assert "localhost" in str(server_cert.subject)
        assert server_cert.issuer.rfc4514_string() == "CN=Federated Learning CA"
    
    def test_generate_client_certificate(self, temp_directory):
        """Test client certificate generation."""
        cert_manager = TLSCertificateManager(cert_dir=temp_directory)
        
        # First generate CA certificate
        cert_manager.generate_ca_certificate()
        
        # Generate client certificate
        client_id = "test_client_001"
        client_cert, client_key = cert_manager.generate_client_certificate(client_id)
        
        # Verify certificate files are created
        client_cert_path = temp_directory / f"{client_id}_cert.pem"
        client_key_path = temp_directory / f"{client_id}_key.pem"
        assert client_cert_path.exists()
        assert client_key_path.exists()
        
        # Verify certificate properties
        assert client_id in str(client_cert.subject)
        assert client_cert.issuer.rfc4514_string() == "CN=Federated Learning CA"
    
    def test_create_ssl_context(self, temp_directory):
        """Test SSL context creation."""
        cert_manager = TLSCertificateManager(cert_dir=temp_directory)
        
        # Generate certificates
        cert_manager.generate_ca_certificate()
        cert_manager.generate_server_certificate("localhost")
        
        # Create SSL context
        ssl_context = cert_manager.create_ssl_context(is_server=True)
        
        # Verify SSL context properties
        assert isinstance(ssl_context, ssl.SSLContext)
        assert ssl_context.protocol == ssl.PROTOCOL_TLS_SERVER
        assert ssl_context.verify_mode == ssl.CERT_REQUIRED
    
    def test_certificate_validation(self, temp_directory):
        """Test certificate validation functionality."""
        cert_manager = TLSCertificateManager(cert_dir=temp_directory)
        
        # Generate certificates
        ca_cert, ca_key = cert_manager.generate_ca_certificate()
        server_cert, server_key = cert_manager.generate_server_certificate("localhost")
        
        # Test certificate validation
        is_valid = cert_manager.validate_certificate(
            cert_manager.ca_cert_path,
            cert_manager.ca_cert_path  # Self-signed CA
        )
        assert is_valid
    
    def test_certificate_expiry_check(self, temp_directory):
        """Test certificate expiry checking."""
        cert_manager = TLSCertificateManager(cert_dir=temp_directory)
        
        # Generate certificate
        ca_cert, ca_key = cert_manager.generate_ca_certificate()
        
        # Check expiry (should be valid for 365 days)
        days_until_expiry = cert_manager.check_certificate_expiry(cert_manager.ca_cert_path)
        assert 360 <= days_until_expiry <= 365  # Allow some margin for test execution time


class TestSecureTokenManager:
    """Test JWT token management functionality."""
    
    def test_token_manager_initialization(self):
        """Test token manager initialization."""
        secret_key = "test_secret_key"
        token_manager = SecureTokenManager(secret_key)
        
        assert token_manager.secret_key == secret_key
        assert token_manager.algorithm == "HS256"
        assert token_manager.token_expiry_hours == 24
    
    def test_generate_token(self):
        """Test JWT token generation."""
        token_manager = SecureTokenManager("test_secret")
        
        client_id = "test_client_001"
        permissions = ["read", "write", "train"]
        
        token = token_manager.generate_token(client_id, permissions)
        
        # Verify token is a string
        assert isinstance(token, str)
        assert len(token) > 0
        
        # Decode and verify token content
        decoded = jwt.decode(token, "test_secret", algorithms=["HS256"])
        assert decoded["client_id"] == client_id
        assert decoded["permissions"] == permissions
        assert "exp" in decoded
        assert "iat" in decoded
    
    def test_validate_token_valid(self):
        """Test validation of valid JWT token."""
        token_manager = SecureTokenManager("test_secret")
        
        client_id = "test_client_001"
        permissions = ["read", "write"]
        token = token_manager.generate_token(client_id, permissions)
        
        # Validate token
        is_valid, payload = token_manager.validate_token(token)
        
        assert is_valid
        assert payload["client_id"] == client_id
        assert payload["permissions"] == permissions
    
    def test_validate_token_expired(self):
        """Test validation of expired JWT token."""
        token_manager = SecureTokenManager("test_secret", token_expiry_hours=0)
        
        # Generate token that expires immediately
        token = token_manager.generate_token("test_client", ["read"])
        
        # Wait a moment and validate (should be expired)
        import time
        time.sleep(0.1)
        
        is_valid, payload = token_manager.validate_token(token)
        
        assert not is_valid
        assert payload is None
    
    def test_validate_token_invalid_signature(self):
        """Test validation of token with invalid signature."""
        token_manager = SecureTokenManager("test_secret")
        
        # Generate token with one secret
        token = token_manager.generate_token("test_client", ["read"])
        
        # Try to validate with different secret
        token_manager_wrong = SecureTokenManager("wrong_secret")
        is_valid, payload = token_manager_wrong.validate_token(token)
        
        assert not is_valid
        assert payload is None
    
    def test_check_permission(self):
        """Test permission checking functionality."""
        token_manager = SecureTokenManager("test_secret")
        
        permissions = ["read", "write", "train"]
        token = token_manager.generate_token("test_client", permissions)
        
        # Test valid permissions
        assert token_manager.check_permission(token, "read")
        assert token_manager.check_permission(token, "write")
        assert token_manager.check_permission(token, "train")
        
        # Test invalid permission
        assert not token_manager.check_permission(token, "admin")
    
    def test_refresh_token(self):
        """Test token refresh functionality."""
        token_manager = SecureTokenManager("test_secret")
        
        original_token = token_manager.generate_token("test_client", ["read"])
        
        # Refresh token
        new_token = token_manager.refresh_token(original_token)
        
        # Verify new token is different but valid
        assert new_token != original_token
        
        is_valid, payload = token_manager.validate_token(new_token)
        assert is_valid
        assert payload["client_id"] == "test_client"
        assert payload["permissions"] == ["read"]


class TestSecureFederatedStrategy:
    """Test secure federated learning strategy with differential privacy."""
    
    def test_strategy_initialization(self):
        """Test secure federated strategy initialization."""
        strategy = SecureFederatedStrategy(
            noise_multiplier=0.1,
            l2_norm_clip=1.0,
            enable_secure_aggregation=True
        )
        
        assert strategy.noise_multiplier == 0.1
        assert strategy.l2_norm_clip == 1.0
        assert strategy.enable_secure_aggregation
    
    def test_differential_privacy_noise_addition(self):
        """Test differential privacy noise addition."""
        strategy = SecureFederatedStrategy(noise_multiplier=0.1, l2_norm_clip=1.0)
        
        # Create mock parameters
        original_params = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0])]
        
        # Apply differential privacy
        noisy_params = strategy.add_differential_privacy_noise(original_params)
        
        # Verify noise was added (parameters should be different)
        assert len(noisy_params) == len(original_params)
        for orig, noisy in zip(original_params, noisy_params):
            assert orig.shape == noisy.shape
            # Parameters should be different due to noise
            assert not np.allclose(orig, noisy, atol=1e-10)
    
    def test_l2_norm_clipping(self):
        """Test L2 norm clipping functionality."""
        strategy = SecureFederatedStrategy(l2_norm_clip=1.0)
        
        # Create parameters with large norm
        large_params = [np.array([10.0, 20.0, 30.0])]
        
        # Apply clipping
        clipped_params = strategy.clip_parameters_by_norm(large_params)
        
        # Verify L2 norm is clipped
        l2_norm = np.sqrt(sum(np.sum(p**2) for p in clipped_params))
        assert l2_norm <= 1.0 + 1e-6  # Allow small numerical error
    
    @patch('flwr.server.strategy.FedAvg.aggregate_fit')
    def test_secure_aggregation(self, mock_aggregate_fit):
        """Test secure aggregation with differential privacy."""
        strategy = SecureFederatedStrategy(
            noise_multiplier=0.1,
            l2_norm_clip=1.0,
            enable_secure_aggregation=True
        )
        
        # Mock aggregation result
        mock_params = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        mock_aggregate_fit.return_value = (mock_params, {})
        
        # Mock fit results
        from flwr.common import FitRes, Parameters, parameters_to_ndarrays, ndarrays_to_parameters
        
        fit_results = [
            (Mock(), FitRes(
                status=Mock(),
                parameters=ndarrays_to_parameters([np.array([1.1, 2.1]), np.array([3.1, 4.1])]),
                num_examples=100,
                metrics={}
            ))
        ]
        
        # Test aggregation
        result_params, result_metrics = strategy.aggregate_fit(1, fit_results, {})
        
        # Verify aggregation was called
        mock_aggregate_fit.assert_called_once()
        
        # Verify result structure
        assert result_params is not None
        assert isinstance(result_metrics, dict)
    
    def test_privacy_budget_tracking(self):
        """Test privacy budget tracking functionality."""
        strategy = SecureFederatedStrategy(noise_multiplier=0.1)
        
        # Initial privacy budget should be 0
        assert strategy.privacy_budget_used == 0.0
        
        # Simulate privacy budget consumption
        strategy.update_privacy_budget(0.5)
        assert strategy.privacy_budget_used == 0.5
        
        # Check if budget is exceeded
        strategy.update_privacy_budget(0.6)
        assert strategy.privacy_budget_used == 1.1
        assert strategy.is_privacy_budget_exceeded(max_budget=1.0)


class TestSecurityAuditor:
    """Test security audit functionality."""
    
    def test_auditor_initialization(self, temp_directory):
        """Test security auditor initialization."""
        auditor = SecurityAuditor(project_root=temp_directory)
        
        assert auditor.project_root == temp_directory
        assert auditor.results == {}
    
    @patch('subprocess.run')
    def test_dependency_vulnerability_scan(self, mock_subprocess, temp_directory):
        """Test dependency vulnerability scanning."""
        auditor = SecurityAuditor(project_root=temp_directory)
        
        # Mock pip-audit output
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({
            "vulnerabilities": [
                {
                    "package": "test-package",
                    "version": "1.0.0",
                    "id": "VULN-001",
                    "description": "Test vulnerability"
                }
            ]
        })
        mock_subprocess.return_value = mock_result
        
        # Run vulnerability scan
        vulnerabilities = auditor.scan_dependencies()
        
        # Verify scan results
        assert len(vulnerabilities) == 1
        assert vulnerabilities[0]["package"] == "test-package"
        assert vulnerabilities[0]["id"] == "VULN-001"
    
    @patch('subprocess.run')
    def test_code_security_analysis(self, mock_subprocess, temp_directory):
        """Test code security analysis with Bandit."""
        auditor = SecurityAuditor(project_root=temp_directory)
        
        # Mock bandit output
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({
            "results": [
                {
                    "filename": "test_file.py",
                    "issue_severity": "HIGH",
                    "issue_confidence": "HIGH",
                    "test_id": "B101",
                    "test_name": "assert_used"
                }
            ]
        })
        mock_subprocess.return_value = mock_result
        
        # Run code analysis
        issues = auditor.analyze_code_security()
        
        # Verify analysis results
        assert len(issues) == 1
        assert issues[0]["test_id"] == "B101"
        assert issues[0]["issue_severity"] == "HIGH"
    
    def test_certificate_validation_audit(self, temp_directory, mock_certificates):
        """Test certificate validation in security audit."""
        auditor = SecurityAuditor(project_root=temp_directory)
        
        # Run certificate validation
        cert_status = auditor.validate_certificates(mock_certificates)
        
        # Verify certificate validation results
        assert isinstance(cert_status, dict)
        assert "ca_cert" in cert_status
        assert "server_cert" in cert_status
        assert "client_cert" in cert_status
    
    def test_comprehensive_security_audit(self, temp_directory):
        """Test comprehensive security audit execution."""
        auditor = SecurityAuditor(project_root=temp_directory)
        
        with patch.object(auditor, 'scan_dependencies') as mock_deps, \
             patch.object(auditor, 'analyze_code_security') as mock_code, \
             patch.object(auditor, 'validate_certificates') as mock_certs:
            
            # Mock audit results
            mock_deps.return_value = []
            mock_code.return_value = []
            mock_certs.return_value = {"status": "valid"}
            
            # Run comprehensive audit
            audit_results = auditor.run_comprehensive_audit()
            
            # Verify all audit components were called
            mock_deps.assert_called_once()
            mock_code.assert_called_once()
            mock_certs.assert_called_once()
            
            # Verify audit results structure
            assert "dependencies" in audit_results
            assert "code_security" in audit_results
            assert "certificates" in audit_results
            assert "timestamp" in audit_results
    
    def test_audit_report_generation(self, temp_directory):
        """Test security audit report generation."""
        auditor = SecurityAuditor(project_root=temp_directory)
        
        # Set mock audit results
        auditor.results = {
            "dependencies": [],
            "code_security": [],
            "certificates": {"status": "valid"},
            "timestamp": datetime.now().isoformat()
        }
        
        # Generate report
        report_path = temp_directory / "security_audit_report.json"
        auditor.generate_report(report_path)
        
        # Verify report file was created
        assert report_path.exists()
        
        # Verify report content
        with open(report_path) as f:
            report_data = json.load(f)
        
        assert "dependencies" in report_data
        assert "code_security" in report_data
        assert "certificates" in report_data
        assert "summary" in report_data


class TestSecurityIntegration:
    """Integration tests for security framework components."""
    
    def test_tls_jwt_integration(self, temp_directory):
        """Test integration between TLS and JWT authentication."""
        # Initialize components
        cert_manager = TLSCertificateManager(cert_dir=temp_directory)
        token_manager = SecureTokenManager("test_secret")
        
        # Generate certificates
        cert_manager.generate_ca_certificate()
        cert_manager.generate_server_certificate("localhost")
        
        # Generate JWT token
        token = token_manager.generate_token("test_client", ["read", "write"])
        
        # Verify both components work together
        ssl_context = cert_manager.create_ssl_context(is_server=True)
        is_valid, payload = token_manager.validate_token(token)
        
        assert ssl_context is not None
        assert is_valid
        assert payload["client_id"] == "test_client"
    
    def test_federated_strategy_security_integration(self, temp_directory):
        """Test integration between federated strategy and security components."""
        # Initialize components
        strategy = SecureFederatedStrategy(
            noise_multiplier=0.1,
            l2_norm_clip=1.0,
            enable_secure_aggregation=True
        )
        token_manager = SecureTokenManager("test_secret")
        
        # Generate token for client
        token = token_manager.generate_token("test_client", ["train"])
        
        # Verify client has training permission
        has_permission = token_manager.check_permission(token, "train")
        assert has_permission
        
        # Test differential privacy with mock parameters
        params = [np.array([1.0, 2.0, 3.0])]
        noisy_params = strategy.add_differential_privacy_noise(params)
        
        # Verify noise was added
        assert not np.allclose(params[0], noisy_params[0], atol=1e-10)
    
    def test_end_to_end_security_workflow(self, temp_directory):
        """Test complete end-to-end security workflow."""
        # 1. Initialize all security components
        cert_manager = TLSCertificateManager(cert_dir=temp_directory)
        token_manager = SecureTokenManager("test_secret")
        strategy = SecureFederatedStrategy(noise_multiplier=0.1, l2_norm_clip=1.0)
        auditor = SecurityAuditor(project_root=temp_directory)
        
        # 2. Generate certificates
        cert_manager.generate_ca_certificate()
        cert_manager.generate_server_certificate("localhost")
        client_cert, client_key = cert_manager.generate_client_certificate("test_client")
        
        # 3. Generate JWT token
        token = token_manager.generate_token("test_client", ["read", "write", "train"])
        
        # 4. Validate security setup
        ssl_context = cert_manager.create_ssl_context(is_server=True)
        is_valid, payload = token_manager.validate_token(token)
        has_train_permission = token_manager.check_permission(token, "train")
        
        # 5. Test differential privacy
        mock_params = [np.array([1.0, 2.0, 3.0])]
        private_params = strategy.add_differential_privacy_noise(mock_params)
        
        # 6. Run security audit (mocked)
        with patch.object(auditor, 'scan_dependencies', return_value=[]), \
             patch.object(auditor, 'analyze_code_security', return_value=[]):
            audit_results = auditor.run_comprehensive_audit()
        
        # Verify complete workflow
        assert ssl_context is not None
        assert is_valid
        assert has_train_permission
        assert not np.allclose(mock_params[0], private_params[0], atol=1e-10)
        assert "dependencies" in audit_results
        assert "code_security" in audit_results


# Performance tests for security components
class TestSecurityPerformance:
    """Performance tests for security framework."""
    
    def test_certificate_generation_performance(self, temp_directory, performance_monitor):
        """Test certificate generation performance."""
        cert_manager = TLSCertificateManager(cert_dir=temp_directory)
        
        with performance_monitor as monitor:
            # Generate multiple certificates
            cert_manager.generate_ca_certificate()
            for i in range(5):
                cert_manager.generate_client_certificate(f"client_{i}")
        
        # Verify performance is acceptable
        assert monitor.execution_time < 10.0  # Should complete within 10 seconds
        assert monitor.memory_usage['peak_mb'] < 100  # Should use less than 100MB
    
    def test_jwt_token_performance(self, performance_monitor):
        """Test JWT token generation and validation performance."""
        token_manager = SecureTokenManager("test_secret")
        
        with performance_monitor as monitor:
            # Generate and validate many tokens
            tokens = []
            for i in range(1000):
                token = token_manager.generate_token(f"client_{i}", ["read", "write"])
                tokens.append(token)
            
            # Validate all tokens
            for token in tokens:
                is_valid, payload = token_manager.validate_token(token)
                assert is_valid
        
        # Verify performance is acceptable
        assert monitor.execution_time < 5.0  # Should complete within 5 seconds
        assert monitor.memory_usage['peak_mb'] < 50  # Should use less than 50MB
    
    def test_differential_privacy_performance(self, performance_monitor):
        """Test differential privacy performance."""
        strategy = SecureFederatedStrategy(noise_multiplier=0.1, l2_norm_clip=1.0)
        
        # Create large parameter arrays
        large_params = [np.random.randn(1000, 1000) for _ in range(5)]
        
        with performance_monitor as monitor:
            # Apply differential privacy to large parameters
            noisy_params = strategy.add_differential_privacy_noise(large_params)
        
        # Verify performance is acceptable
        assert monitor.execution_time < 2.0  # Should complete within 2 seconds
        assert len(noisy_params) == len(large_params)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

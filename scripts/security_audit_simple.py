#!/usr/bin/env python3
"""
Simplified Security Audit Script
Raspberry Pi 5 Federated Environmental Monitoring Network

A simplified version of the security audit for Sprint 6 demonstration
that doesn't require external dependencies.
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any


class SimpleSecurityAuditor:
    """Simplified security auditor for demonstration."""
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path(__file__).parent.parent
        self.results = {}
    
    def check_file_permissions(self) -> Dict[str, Any]:
        """Check file permissions for security issues."""
        print("🔍 Checking file permissions...")
        
        issues = []
        
        # Check for world-writable files
        for root, dirs, files in os.walk(self.project_root):
            for file in files:
                file_path = Path(root) / file
                try:
                    stat = file_path.stat()
                    # Check if world-writable (others have write permission)
                    if stat.st_mode & 0o002:
                        issues.append({
                            'file': str(file_path),
                            'issue': 'World-writable file',
                            'permissions': oct(stat.st_mode)[-3:]
                        })
                except (OSError, PermissionError):
                    continue
        
        return {
            'issues_found': len(issues),
            'issues': issues,
            'status': 'PASS' if len(issues) == 0 else 'WARN'
        }
    
    def check_sensitive_files(self) -> Dict[str, Any]:
        """Check for sensitive files that shouldn't be in the repository."""
        print("🔍 Checking for sensitive files...")
        
        sensitive_patterns = [
            '*.key', '*.pem', '*.p12', '*.pfx',
            '.env', '.env.*', 'secrets.*',
            'id_rsa', 'id_dsa', 'id_ecdsa',
            'config.json', 'credentials.*'
        ]
        
        found_files = []
        
        for pattern in sensitive_patterns:
            for file_path in self.project_root.rglob(pattern):
                if not any(part.startswith('.git') for part in file_path.parts):
                    found_files.append({
                        'file': str(file_path.relative_to(self.project_root)),
                        'pattern': pattern,
                        'size': file_path.stat().st_size if file_path.exists() else 0
                    })
        
        return {
            'files_found': len(found_files),
            'files': found_files,
            'status': 'PASS' if len(found_files) == 0 else 'WARN'
        }
    
    def check_python_imports(self) -> Dict[str, Any]:
        """Check Python files for potentially dangerous imports."""
        print("🔍 Checking Python imports...")
        
        dangerous_imports = [
            'eval', 'exec', 'compile', '__import__',
            'subprocess.call', 'os.system', 'commands.',
            'pickle.loads', 'cPickle.loads', 'marshal.loads'
        ]
        
        issues = []
        
        for py_file in self.project_root.rglob('*.py'):
            if any(part.startswith('.') for part in py_file.parts):
                continue
            
            try:
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
                for line_num, line in enumerate(content.splitlines(), 1):
                    for dangerous in dangerous_imports:
                        if dangerous in line and not line.strip().startswith('#'):
                            issues.append({
                                'file': str(py_file.relative_to(self.project_root)),
                                'line': line_num,
                                'issue': f'Potentially dangerous import: {dangerous}',
                                'code': line.strip()
                            })
            except (OSError, UnicodeDecodeError):
                continue
        
        return {
            'issues_found': len(issues),
            'issues': issues,
            'status': 'PASS' if len(issues) == 0 else 'WARN'
        }
    
    def check_hardcoded_secrets(self) -> Dict[str, Any]:
        """Check for hardcoded secrets in code."""
        print("🔍 Checking for hardcoded secrets...")
        
        secret_patterns = [
            ('password', r'password\s*=\s*["\'][^"\']+["\']'),
            ('api_key', r'api_key\s*=\s*["\'][^"\']+["\']'),
            ('secret', r'secret\s*=\s*["\'][^"\']+["\']'),
            ('token', r'token\s*=\s*["\'][^"\']+["\']'),
        ]
        
        issues = []
        
        for py_file in self.project_root.rglob('*.py'):
            if any(part.startswith('.') for part in py_file.parts):
                continue
            
            try:
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
                for line_num, line in enumerate(content.splitlines(), 1):
                    if line.strip().startswith('#'):
                        continue
                    
                    for secret_type, pattern in secret_patterns:
                        import re
                        if re.search(pattern, line, re.IGNORECASE):
                            # Skip test files and obvious test values
                            if ('test' not in str(py_file).lower() and 
                                'test_' not in line.lower() and
                                'example' not in line.lower() and
                                'dummy' not in line.lower()):
                                issues.append({
                                    'file': str(py_file.relative_to(self.project_root)),
                                    'line': line_num,
                                    'type': secret_type,
                                    'issue': f'Potential hardcoded {secret_type}',
                                })
            except (OSError, UnicodeDecodeError):
                continue
        
        return {
            'issues_found': len(issues),
            'issues': issues,
            'status': 'PASS' if len(issues) == 0 else 'WARN'
        }
    
    def check_security_framework(self) -> Dict[str, Any]:
        """Check if security framework components are present."""
        print("🔍 Checking security framework implementation...")
        
        required_files = [
            'server/security/tls_config.py',
            'server/security/secure_federated_server.py',
            'SECURITY_GUIDE.md',
            'THIRD_PARTY_LICENSES.md'
        ]
        
        missing_files = []
        present_files = []
        
        for file_path in required_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                present_files.append(file_path)
            else:
                missing_files.append(file_path)
        
        # Check for security-related classes
        security_classes = []
        tls_config_path = self.project_root / 'server/security/tls_config.py'
        if tls_config_path.exists():
            try:
                with open(tls_config_path, 'r') as f:
                    content = f.read()
                    if 'class TLSCertificateManager' in content:
                        security_classes.append('TLSCertificateManager')
                    if 'class SecureTokenManager' in content:
                        security_classes.append('SecureTokenManager')
            except OSError:
                pass
        
        return {
            'required_files': len(required_files),
            'present_files': len(present_files),
            'missing_files': missing_files,
            'security_classes': security_classes,
            'status': 'PASS' if len(missing_files) == 0 else 'FAIL'
        }
    
    def run_comprehensive_audit(self) -> Dict[str, Any]:
        """Run comprehensive security audit."""
        print("🚀 Starting comprehensive security audit...")
        print(f"📁 Project root: {self.project_root}")
        
        audit_results = {
            'timestamp': datetime.now().isoformat(),
            'project_root': str(self.project_root),
            'audits': {
                'file_permissions': self.check_file_permissions(),
                'sensitive_files': self.check_sensitive_files(),
                'python_imports': self.check_python_imports(),
                'hardcoded_secrets': self.check_hardcoded_secrets(),
                'security_framework': self.check_security_framework(),
            }
        }
        
        # Calculate overall status
        statuses = [audit['status'] for audit in audit_results['audits'].values()]
        if 'FAIL' in statuses:
            overall_status = 'FAIL'
        elif 'WARN' in statuses:
            overall_status = 'WARN'
        else:
            overall_status = 'PASS'
        
        audit_results['overall_status'] = overall_status
        audit_results['summary'] = self._generate_summary(audit_results)
        
        self.results = audit_results
        return audit_results
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate audit summary."""
        total_issues = sum(
            audit.get('issues_found', 0) + audit.get('files_found', 0)
            for audit in results['audits'].values()
        )
        
        return {
            'total_audits': len(results['audits']),
            'total_issues': total_issues,
            'passed_audits': sum(1 for audit in results['audits'].values() if audit['status'] == 'PASS'),
            'warning_audits': sum(1 for audit in results['audits'].values() if audit['status'] == 'WARN'),
            'failed_audits': sum(1 for audit in results['audits'].values() if audit['status'] == 'FAIL'),
        }
    
    def save_results(self, output_file: str):
        """Save audit results to file."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"💾 Audit results saved to: {output_file}")
    
    def print_summary(self):
        """Print audit summary."""
        if not self.results:
            print("❌ No audit results available")
            return
        
        summary = self.results['summary']
        overall_status = self.results['overall_status']
        
        print("\n" + "="*60)
        print("🔒 SECURITY AUDIT SUMMARY")
        print("="*60)
        
        status_emoji = {
            'PASS': '✅',
            'WARN': '⚠️',
            'FAIL': '❌'
        }
        
        print(f"\n{status_emoji.get(overall_status, '❓')} Overall Status: {overall_status}")
        print(f"📊 Total Audits: {summary['total_audits']}")
        print(f"🔍 Total Issues: {summary['total_issues']}")
        print(f"✅ Passed: {summary['passed_audits']}")
        print(f"⚠️  Warnings: {summary['warning_audits']}")
        print(f"❌ Failed: {summary['failed_audits']}")
        
        print("\n📋 Audit Details:")
        for audit_name, audit_result in self.results['audits'].items():
            status = audit_result['status']
            emoji = status_emoji.get(status, '❓')
            print(f"  {emoji} {audit_name.replace('_', ' ').title()}: {status}")
            
            if audit_result.get('issues_found', 0) > 0:
                print(f"    └─ Issues: {audit_result['issues_found']}")
            if audit_result.get('files_found', 0) > 0:
                print(f"    └─ Files: {audit_result['files_found']}")
        
        # Security Framework Status
        sf_result = self.results['audits']['security_framework']
        print(f"\n🛡️  Security Framework Status:")
        print(f"  📁 Required files: {sf_result['required_files']}")
        print(f"  ✅ Present files: {sf_result['present_files']}")
        print(f"  🔧 Security classes: {len(sf_result['security_classes'])}")
        
        if sf_result['security_classes']:
            print(f"    └─ Classes: {', '.join(sf_result['security_classes'])}")
        
        if sf_result['missing_files']:
            print(f"  ❌ Missing files: {len(sf_result['missing_files'])}")
            for missing in sf_result['missing_files']:
                print(f"    └─ {missing}")
        
        print("\n" + "="*60)


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Simplified Security Audit Script")
    parser.add_argument('--output-file', '-o', type=str, help='Output file for results')
    parser.add_argument('--output-format', type=str, choices=['json', 'text'], 
                       default='text', help='Output format')
    
    args = parser.parse_args()
    
    # Run security audit
    auditor = SimpleSecurityAuditor()
    results = auditor.run_comprehensive_audit()
    
    # Print summary
    auditor.print_summary()
    
    # Save results if requested
    if args.output_file:
        auditor.save_results(args.output_file)
    
    # Exit with appropriate code
    overall_status = results['overall_status']
    if overall_status == 'FAIL':
        return 1
    elif overall_status == 'WARN':
        return 0  # Warnings don't fail the build
    else:
        return 0


if __name__ == "__main__":
    sys.exit(main())

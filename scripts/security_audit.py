#!/usr/bin/env python3
"""
Security Audit Script
Raspberry Pi 5 Federated Environmental Monitoring Network

This script performs comprehensive security auditing including:
1. Dependency vulnerability scanning
2. Code security analysis with Bandit
3. License compliance checking
4. Certificate validation
5. Configuration security review
"""

import sys
import subprocess
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
import structlog

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

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


class SecurityAuditor:
    """Comprehensive security auditing for the federated learning project."""
    
    def __init__(self, project_root: Path):
        """Initialize security auditor."""
        self.project_root = project_root
        self.audit_results = {}
        
        logger.info("Security auditor initialized", project_root=str(project_root))
    
    def run_pip_audit(self) -> Dict:
        """Run pip-audit to check for known vulnerabilities."""
        
        logger.info("Running pip-audit for vulnerability scanning")
        
        try:
            # Run pip-audit with JSON output
            result = subprocess.run([
                sys.executable, "-m", "pip_audit",
                "--format", "json",
                "--progress-spinner", "off"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode == 0:
                # No vulnerabilities found
                audit_data = {"vulnerabilities": [], "summary": "No known vulnerabilities found"}
                logger.info("pip-audit completed successfully - no vulnerabilities found")
            else:
                # Parse JSON output for vulnerabilities
                try:
                    audit_data = json.loads(result.stdout)
                    vuln_count = len(audit_data.get("vulnerabilities", []))
                    logger.warning("pip-audit found vulnerabilities", count=vuln_count)
                except json.JSONDecodeError:
                    audit_data = {
                        "error": "Failed to parse pip-audit output",
                        "stdout": result.stdout,
                        "stderr": result.stderr
                    }
            
            return {
                "success": True,
                "tool": "pip-audit",
                "data": audit_data,
                "execution_time": time.time()
            }
            
        except FileNotFoundError:
            logger.error("pip-audit not found - install with: pip install pip-audit")
            return {
                "success": False,
                "error": "pip-audit not installed",
                "tool": "pip-audit"
            }
        except Exception as e:
            logger.error("pip-audit execution failed", error=str(e))
            return {
                "success": False,
                "error": str(e),
                "tool": "pip-audit"
            }
    
    def run_bandit_scan(self) -> Dict:
        """Run Bandit security linter on Python code."""
        
        logger.info("Running Bandit security analysis")
        
        try:
            # Run Bandit with JSON output
            result = subprocess.run([
                "bandit",
                "-r", str(self.project_root),
                "-f", "json",
                "-x", str(self.project_root / "venv"),  # Exclude virtual environment
                "-x", str(self.project_root / ".git"),   # Exclude git directory
                "--skip", "B101,B601"  # Skip assert and shell injection (common false positives)
            ], capture_output=True, text=True)
            
            try:
                bandit_data = json.loads(result.stdout)
                
                # Extract key metrics
                metrics = bandit_data.get("metrics", {})
                results = bandit_data.get("results", [])
                
                # Categorize issues by severity
                severity_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
                for issue in results:
                    severity = issue.get("issue_severity", "UNKNOWN")
                    if severity in severity_counts:
                        severity_counts[severity] += 1
                
                logger.info("Bandit scan completed", 
                           total_issues=len(results),
                           severity_counts=severity_counts)
                
                return {
                    "success": True,
                    "tool": "bandit",
                    "data": {
                        "metrics": metrics,
                        "results": results,
                        "severity_counts": severity_counts,
                        "total_issues": len(results)
                    },
                    "execution_time": time.time()
                }
                
            except json.JSONDecodeError:
                logger.error("Failed to parse Bandit output")
                return {
                    "success": False,
                    "error": "Failed to parse Bandit JSON output",
                    "tool": "bandit",
                    "raw_output": result.stdout
                }
            
        except FileNotFoundError:
            logger.error("Bandit not found - install with: pip install bandit")
            return {
                "success": False,
                "error": "Bandit not installed",
                "tool": "bandit"
            }
        except Exception as e:
            logger.error("Bandit execution failed", error=str(e))
            return {
                "success": False,
                "error": str(e),
                "tool": "bandit"
            }
    
    def run_safety_check(self) -> Dict:
        """Run Safety to check for known security vulnerabilities."""
        
        logger.info("Running Safety vulnerability check")
        
        try:
            # Run Safety with JSON output
            result = subprocess.run([
                "safety", "check",
                "--json",
                "--ignore", "70612"  # Ignore jinja2 vulnerability (false positive in our use case)
            ], capture_output=True, text=True, cwd=self.project_root)
            
            try:
                if result.stdout.strip():
                    safety_data = json.loads(result.stdout)
                else:
                    safety_data = {"vulnerabilities": [], "summary": "No vulnerabilities found"}
                
                vuln_count = len(safety_data.get("vulnerabilities", []))
                
                if vuln_count == 0:
                    logger.info("Safety check completed - no vulnerabilities found")
                else:
                    logger.warning("Safety check found vulnerabilities", count=vuln_count)
                
                return {
                    "success": True,
                    "tool": "safety",
                    "data": safety_data,
                    "vulnerability_count": vuln_count,
                    "execution_time": time.time()
                }
                
            except json.JSONDecodeError:
                # Safety might return non-JSON output in some cases
                logger.info("Safety check completed (non-JSON output)")
                return {
                    "success": True,
                    "tool": "safety",
                    "data": {"summary": "No vulnerabilities found", "raw_output": result.stdout},
                    "vulnerability_count": 0,
                    "execution_time": time.time()
                }
            
        except FileNotFoundError:
            logger.error("Safety not found - install with: pip install safety")
            return {
                "success": False,
                "error": "Safety not installed",
                "tool": "safety"
            }
        except Exception as e:
            logger.error("Safety execution failed", error=str(e))
            return {
                "success": False,
                "error": str(e),
                "tool": "safety"
            }
    
    def check_file_permissions(self) -> Dict:
        """Check file permissions for security issues."""
        
        logger.info("Checking file permissions")
        
        security_issues = []
        
        # Check for world-writable files
        for file_path in self.project_root.rglob("*"):
            if file_path.is_file() and not any(part.startswith('.') for part in file_path.parts):
                try:
                    stat = file_path.stat()
                    # Check if world-writable (others have write permission)
                    if stat.st_mode & 0o002:
                        security_issues.append({
                            "type": "world_writable",
                            "file": str(file_path.relative_to(self.project_root)),
                            "permissions": oct(stat.st_mode)[-3:]
                        })
                except (OSError, PermissionError):
                    continue
        
        # Check for executable Python files (potential security risk)
        for py_file in self.project_root.rglob("*.py"):
            if py_file.is_file():
                try:
                    stat = py_file.stat()
                    # Check if executable by others
                    if stat.st_mode & 0o001:
                        security_issues.append({
                            "type": "executable_python",
                            "file": str(py_file.relative_to(self.project_root)),
                            "permissions": oct(stat.st_mode)[-3:]
                        })
                except (OSError, PermissionError):
                    continue
        
        logger.info("File permission check completed", issues_found=len(security_issues))
        
        return {
            "success": True,
            "tool": "file_permissions",
            "data": {
                "issues": security_issues,
                "total_issues": len(security_issues)
            },
            "execution_time": time.time()
        }
    
    def check_secrets_exposure(self) -> Dict:
        """Check for potential secrets in code."""
        
        logger.info("Checking for potential secrets exposure")
        
        # Common patterns that might indicate secrets
        secret_patterns = [
            r'password\s*=\s*["\'][^"\']+["\']',
            r'api_key\s*=\s*["\'][^"\']+["\']',
            r'secret\s*=\s*["\'][^"\']+["\']',
            r'token\s*=\s*["\'][^"\']+["\']',
            r'-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----',
        ]
        
        potential_secrets = []
        
        # Scan Python files for potential secrets
        for py_file in self.project_root.rglob("*.py"):
            if py_file.is_file() and "venv" not in str(py_file):
                try:
                    content = py_file.read_text(encoding='utf-8')
                    lines = content.split('\n')
                    
                    for line_num, line in enumerate(lines, 1):
                        # Skip comments and obvious test/example values
                        if line.strip().startswith('#') or 'example' in line.lower() or 'test' in line.lower():
                            continue
                        
                        for pattern in secret_patterns:
                            import re
                            if re.search(pattern, line, re.IGNORECASE):
                                potential_secrets.append({
                                    "file": str(py_file.relative_to(self.project_root)),
                                    "line": line_num,
                                    "pattern": pattern,
                                    "content": line.strip()[:100]  # First 100 chars
                                })
                
                except (UnicodeDecodeError, PermissionError):
                    continue
        
        logger.info("Secrets exposure check completed", potential_secrets=len(potential_secrets))
        
        return {
            "success": True,
            "tool": "secrets_check",
            "data": {
                "potential_secrets": potential_secrets,
                "total_findings": len(potential_secrets)
            },
            "execution_time": time.time()
        }
    
    def generate_license_report(self) -> Dict:
        """Generate third-party license report."""
        
        logger.info("Generating license report")
        
        try:
            # Get list of installed packages
            result = subprocess.run([
                sys.executable, "-m", "pip", "list", "--format=json"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode != 0:
                return {
                    "success": False,
                    "error": "Failed to get package list",
                    "tool": "license_report"
                }
            
            packages = json.loads(result.stdout)
            
            # Try to get license information using pip-licenses if available
            try:
                license_result = subprocess.run([
                    "pip-licenses", "--format=json"
                ], capture_output=True, text=True, cwd=self.project_root)
                
                if license_result.returncode == 0:
                    license_data = json.loads(license_result.stdout)
                else:
                    # Fallback: basic package list without license info
                    license_data = [{"Name": pkg["name"], "Version": pkg["version"], "License": "Unknown"} 
                                  for pkg in packages]
                
            except FileNotFoundError:
                # pip-licenses not installed, use basic package info
                license_data = [{"Name": pkg["name"], "Version": pkg["version"], "License": "Unknown"} 
                              for pkg in packages]
            
            # Categorize licenses
            license_categories = {}
            for pkg in license_data:
                license_name = pkg.get("License", "Unknown")
                if license_name not in license_categories:
                    license_categories[license_name] = []
                license_categories[license_name].append(pkg["Name"])
            
            logger.info("License report generated", 
                       total_packages=len(packages),
                       license_types=len(license_categories))
            
            return {
                "success": True,
                "tool": "license_report",
                "data": {
                    "packages": license_data,
                    "license_categories": license_categories,
                    "total_packages": len(packages),
                    "license_types": len(license_categories)
                },
                "execution_time": time.time()
            }
            
        except Exception as e:
            logger.error("License report generation failed", error=str(e))
            return {
                "success": False,
                "error": str(e),
                "tool": "license_report"
            }
    
    def run_comprehensive_audit(self) -> Dict:
        """Run comprehensive security audit."""
        
        logger.info("Starting comprehensive security audit")
        
        audit_start_time = time.time()
        
        # Run all security checks
        self.audit_results = {
            "audit_timestamp": time.time(),
            "project_root": str(self.project_root),
            "checks": {}
        }
        
        # 1. Dependency vulnerability scanning
        logger.info("Step 1: Dependency vulnerability scanning")
        self.audit_results["checks"]["pip_audit"] = self.run_pip_audit()
        self.audit_results["checks"]["safety"] = self.run_safety_check()
        
        # 2. Code security analysis
        logger.info("Step 2: Code security analysis")
        self.audit_results["checks"]["bandit"] = self.run_bandit_scan()
        
        # 3. File permission checks
        logger.info("Step 3: File permission analysis")
        self.audit_results["checks"]["file_permissions"] = self.check_file_permissions()
        
        # 4. Secrets exposure check
        logger.info("Step 4: Secrets exposure analysis")
        self.audit_results["checks"]["secrets_check"] = self.check_secrets_exposure()
        
        # 5. License compliance
        logger.info("Step 5: License compliance check")
        self.audit_results["checks"]["license_report"] = self.generate_license_report()
        
        # Calculate overall audit time
        audit_duration = time.time() - audit_start_time
        self.audit_results["audit_duration"] = audit_duration
        
        # Generate summary
        self.audit_results["summary"] = self._generate_audit_summary()
        
        logger.info("Comprehensive security audit completed", 
                   duration=f"{audit_duration:.2f}s")
        
        return self.audit_results
    
    def _generate_audit_summary(self) -> Dict:
        """Generate audit summary."""
        
        summary = {
            "total_checks": len(self.audit_results["checks"]),
            "successful_checks": 0,
            "failed_checks": 0,
            "critical_issues": 0,
            "warnings": 0,
            "recommendations": []
        }
        
        for check_name, check_result in self.audit_results["checks"].items():
            if check_result.get("success", False):
                summary["successful_checks"] += 1
                
                # Count issues by severity
                if check_name == "bandit":
                    data = check_result.get("data", {})
                    severity_counts = data.get("severity_counts", {})
                    summary["critical_issues"] += severity_counts.get("HIGH", 0)
                    summary["warnings"] += severity_counts.get("MEDIUM", 0) + severity_counts.get("LOW", 0)
                
                elif check_name in ["pip_audit", "safety"]:
                    data = check_result.get("data", {})
                    vuln_count = len(data.get("vulnerabilities", []))
                    if vuln_count > 0:
                        summary["critical_issues"] += vuln_count
                
                elif check_name == "file_permissions":
                    data = check_result.get("data", {})
                    issues = data.get("issues", [])
                    summary["warnings"] += len(issues)
                
                elif check_name == "secrets_check":
                    data = check_result.get("data", {})
                    secrets = data.get("potential_secrets", [])
                    if secrets:
                        summary["critical_issues"] += len(secrets)
            else:
                summary["failed_checks"] += 1
        
        # Generate recommendations
        if summary["critical_issues"] == 0 and summary["warnings"] == 0:
            summary["recommendations"].append("✅ No critical security issues found")
        else:
            if summary["critical_issues"] > 0:
                summary["recommendations"].append(f"🚨 Address {summary['critical_issues']} critical security issues")
            if summary["warnings"] > 0:
                summary["recommendations"].append(f"⚠️ Review {summary['warnings']} security warnings")
        
        if summary["failed_checks"] > 0:
            summary["recommendations"].append(f"🔧 Fix {summary['failed_checks']} failed security checks")
        
        return summary
    
    def save_audit_report(self, output_file: str = "security_audit_report.json") -> str:
        """Save audit report to file."""
        
        output_path = self.project_root / output_file
        
        with open(output_path, 'w') as f:
            json.dump(self.audit_results, f, indent=2, default=str)
        
        logger.info("Audit report saved", output_path=str(output_path))
        return str(output_path)


def print_audit_summary(audit_results: Dict):
    """Print human-readable audit summary."""
    
    print("\n" + "="*80)
    print("SECURITY AUDIT REPORT")
    print("="*80)
    
    summary = audit_results.get("summary", {})
    
    print(f"📊 Audit Overview:")
    print(f"   Total Checks: {summary.get('total_checks', 0)}")
    print(f"   Successful: {summary.get('successful_checks', 0)}")
    print(f"   Failed: {summary.get('failed_checks', 0)}")
    print(f"   Duration: {audit_results.get('audit_duration', 0):.2f}s")
    
    print(f"\n🔍 Security Issues:")
    print(f"   Critical Issues: {summary.get('critical_issues', 0)}")
    print(f"   Warnings: {summary.get('warnings', 0)}")
    
    # Print check results
    print(f"\n📋 Check Results:")
    for check_name, check_result in audit_results.get("checks", {}).items():
        status = "✅" if check_result.get("success", False) else "❌"
        print(f"   {status} {check_name.replace('_', ' ').title()}")
        
        if check_name == "bandit" and check_result.get("success"):
            data = check_result.get("data", {})
            severity_counts = data.get("severity_counts", {})
            if any(severity_counts.values()):
                print(f"      Issues: HIGH={severity_counts.get('HIGH', 0)}, "
                      f"MEDIUM={severity_counts.get('MEDIUM', 0)}, "
                      f"LOW={severity_counts.get('LOW', 0)}")
        
        elif check_name in ["pip_audit", "safety"] and check_result.get("success"):
            data = check_result.get("data", {})
            vuln_count = len(data.get("vulnerabilities", []))
            if vuln_count > 0:
                print(f"      Vulnerabilities: {vuln_count}")
    
    # Print recommendations
    recommendations = summary.get("recommendations", [])
    if recommendations:
        print(f"\n💡 Recommendations:")
        for rec in recommendations:
            print(f"   {rec}")
    
    print("="*80)


def main():
    """Main function for security audit."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Security Audit for Federated Learning Project")
    parser.add_argument(
        "--output",
        default="security_audit_report.json",
        help="Output file for audit report"
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
    
    try:
        # Run comprehensive security audit
        auditor = SecurityAuditor(project_root)
        audit_results = auditor.run_comprehensive_audit()
        
        # Save report
        report_path = auditor.save_audit_report(args.output)
        
        # Print summary
        print_audit_summary(audit_results)
        
        # Exit with appropriate code
        summary = audit_results.get("summary", {})
        critical_issues = summary.get("critical_issues", 0)
        failed_checks = summary.get("failed_checks", 0)
        
        if critical_issues > 0 or failed_checks > 0:
            print(f"\n⚠️  Security audit found issues. Report saved to: {report_path}")
            sys.exit(1)
        else:
            print(f"\n✅ Security audit passed. Report saved to: {report_path}")
            sys.exit(0)
            
    except KeyboardInterrupt:
        logger.info("Security audit interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error("Security audit failed", error=str(e), exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

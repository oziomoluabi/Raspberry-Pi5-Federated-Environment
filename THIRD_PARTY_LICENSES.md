# Third-Party Licenses

This document contains the licenses of all third-party dependencies used in the Raspberry Pi 5 Federated Environmental Monitoring Network project.

## Summary

This project uses various open-source libraries and frameworks. Below is a comprehensive list of all third-party dependencies and their respective licenses.

## Core Dependencies

### Machine Learning & Federated Learning
- **TensorFlow** (Apache 2.0) - Machine learning framework
- **Flower** (Apache 2.0) - Federated learning framework
- **scikit-learn** (BSD-3-Clause) - Machine learning library
- **NumPy** (BSD-3-Clause) - Numerical computing
- **Pandas** (BSD-3-Clause) - Data manipulation and analysis
- **SciPy** (BSD-3-Clause) - Scientific computing

### Security & Cryptography
- **cryptography** (Apache 2.0 / BSD-3-Clause) - Cryptographic recipes and primitives
- **PyJWT** (MIT) - JSON Web Token implementation
- **bandit** (Apache 2.0) - Security linter for Python
- **safety** (MIT) - Dependency vulnerability scanner
- **pip-audit** (Apache 2.0) - Vulnerability scanner for Python packages

### Hardware & Sensor Integration
- **sense-hat** (BSD-3-Clause) - Raspberry Pi Sense HAT library
- **adafruit-circuitpython-adxl34x** (MIT) - ADXL345 accelerometer library
- **RPi.GPIO** (MIT) - Raspberry Pi GPIO library
- **gpiozero** (BSD-3-Clause) - Simple GPIO library

### MATLAB Integration
- **matlabengine** (Proprietary - MathWorks) - MATLAB Engine for Python
- **oct2py** (MIT) - Python to GNU Octave bridge

### Development & Testing
- **pytest** (MIT) - Testing framework
- **pytest-mock** (MIT) - Mock plugin for pytest
- **structlog** (Apache 2.0 / MIT) - Structured logging
- **black** (MIT) - Code formatter
- **flake8** (MIT) - Code linter

### Utilities
- **PyYAML** (MIT) - YAML parser and emitter
- **python-dotenv** (BSD-3-Clause) - Environment variable loader
- **click** (BSD-3-Clause) - Command line interface creation toolkit
- **psutil** (BSD-3-Clause) - System and process utilities
- **requests** (Apache 2.0) - HTTP library

## License Types Summary

### Apache 2.0 License
- TensorFlow
- Flower
- cryptography (dual licensed)
- bandit
- pip-audit
- structlog (dual licensed)
- requests

### MIT License
- PyJWT
- safety
- adafruit-circuitpython-adxl34x
- RPi.GPIO
- oct2py
- pytest
- pytest-mock
- black
- flake8
- PyYAML

### BSD-3-Clause License
- scikit-learn
- NumPy
- Pandas
- SciPy
- sense-hat
- gpiozero
- python-dotenv
- click
- psutil

### Proprietary License
- matlabengine (MathWorks) - Optional dependency

## License Compliance

This project is compliant with all third-party licenses:

1. **Attribution**: All required attributions are included in this document
2. **Source Code**: Source code is available for all open-source dependencies
3. **License Preservation**: Original license files are preserved in dependency packages
4. **Compatibility**: All licenses are compatible with the project's MIT license

## Optional Dependencies

Some dependencies are optional and only required for specific features:

- **MATLAB Engine for Python**: Required only for MATLAB integration features
- **Hardware-specific libraries**: Required only when running on Raspberry Pi hardware

## Vulnerability Management

This project uses automated tools to monitor dependencies for security vulnerabilities:

- **pip-audit**: Scans for known vulnerabilities in Python packages
- **safety**: Additional vulnerability scanning
- **Dependabot**: Automated dependency updates via GitHub

## Updates and Maintenance

This license document is automatically updated when dependencies change. The project maintains:

- Regular dependency updates for security patches
- Automated vulnerability scanning in CI/CD
- License compatibility verification for new dependencies

## Contact

For questions about licensing or to report license compliance issues, please:

1. Open an issue on the project repository
2. Contact the project maintainers
3. Review the project's CONTRIBUTING.md for contribution guidelines

---

*Last updated: August 2, 2025*
*Generated automatically as part of Sprint 5: Security & Compliance*

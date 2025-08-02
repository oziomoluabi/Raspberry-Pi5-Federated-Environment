# Raspberry Pi 5 Federated Environmental Monitoring Network

[![Build Status](https://github.com/YourOrg/Raspberry-Pi5-Federated/actions/workflows/ci.yml/badge.svg)](https://github.com/YourOrg/Raspberry-Pi5-Federated/actions)
[![Release](https://github.com/YourOrg/Raspberry-Pi5-Federated/actions/workflows/release.yml/badge.svg)](https://github.com/YourOrg/Raspberry-Pi5-Federated/actions/workflows/release.yml)
[![Test Coverage](https://img.shields.io/codecov/c/github/YourOrg/Raspberry-Pi5-Federated.svg)](https://codecov.io/gh/YourOrg/Raspberry-Pi5-Federated)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://python.org)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Security: Bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

[![GitHub issues](https://img.shields.io/github/issues/YourOrg/Raspberry-Pi5-Federated)](https://github.com/YourOrg/Raspberry-Pi5-Federated/issues)
[![GitHub pull requests](https://img.shields.io/github/issues-pr/YourOrg/Raspberry-Pi5-Federated)](https://github.com/YourOrg/Raspberry-Pi5-Federated/pulls)
[![GitHub stars](https://img.shields.io/github/stars/YourOrg/Raspberry-Pi5-Federated)](https://github.com/YourOrg/Raspberry-Pi5-Federated/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/YourOrg/Raspberry-Pi5-Federated)](https://github.com/YourOrg/Raspberry-Pi5-Federated/network)
[![Contributors](https://img.shields.io/github/contributors/YourOrg/Raspberry-Pi5-Federated)](https://github.com/YourOrg/Raspberry-Pi5-Federated/graphs/contributors)  

A fully open-source edge-AI platform on Raspberry Pi that combines:  
- **Federated LSTM forecasting** of environmental data (temperature, humidity)  
- **TinyML autoencoder** for vibration anomaly detection with on-device fine-tuning  
- **MATLAB/Simulink integration** for advanced analytics and code generation  

All development is orchestrated in a **multi-root VS Code** workspace with Dev Containers, Remote-SSH, Jupyter, and automated CI/CD.

---

## 🚀 Key Features

- **Federated Learning**  
  - Raspberry Pi 5 nodes fine-tune an LSTM on local sensor data  
  - Periodic secure weight aggregation with Flower or TensorFlow Federated  
- **TinyML Predictive Maintenance**  
  - On-device autoencoder inference (TFLite-Micro)  
  - Periodic one-step SGD updates on Pi  
- **MATLAB/Simulink Workflows**  
  - MATLAB Engine API calls from Python  
  - Simulink models prototyped and compiled into Python packages  
  - GNU Octave fallback via Oct2Py  

---

## 🧰 Hardware Requirements

- **Raspberry Pi 5 (8 GB)**  
- **Sense HAT** (temp, humidity, pressure, IMU)  
- **SparkFun ADXL345** 3-axis accelerometer breakout (I²C/SPI)  

---

## 🖥️ Software Requirements

- **Python 3.11+**  
- **Flower ≥ 1.4** *or* **TensorFlow Federated ≥ 0.34**  
- **TensorFlow 2.12+**, **Keras**, **TensorFlow Lite**  
- **Apache TVM ≥ 0.10** + **microTVM**  
- **MATLAB® R2022b+** (optional) & **MATLAB Engine for Python**  
- **GNU Octave 7.x** & **Oct2Py** (optional)  
- **Docker**, **Dev Containers**, **CMake ≥ 3.20**, **gcc**, **pybind11**  

---

## 📁 Project Structure

```text
├── server/         # Federated-learning aggregator code
├── client/         # Edge-node scripts: sensing, ML, TinyML, MATLAB hooks
├── matlab/         # .m, .mlx, .slx, codegen outputs
├── docs/           # Architecture diagrams, guides
├── scripts/        # DevOps & utility scripts
├── tests/          # Unit & integration tests
├── .github/        # CI workflows, issue/PR templates
├── .devcontainer/  # Dev Container configuration
├── .vscode/        # VS Code settings, tasks, launch configs
├── README.md
├── LICENSE
└── CONTRIBUTING.md
```

---

## ⚙️ Setup & Quickstart

### 1. Clone & Open Workspace
```bash
git clone https://github.com/YourOrg/Raspberry-Pi5-Federated.git
cd Raspberry-Pi5-Federated
code IoT_Edge.code-workspace
```

### 2. Virtual Environment Setup
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 3. Development Container (Recommended)
- In VS Code: **Remote-Containers: Reopen in Container**  
- All dependencies will be automatically installed

---

## ▶️ Usage

### Run Federated Server & Client
From VS Code **Run Task**:

1. **Run Federated Server**  
2. **Run Edge Client**

### CLI (without VS Code)
```bash
# Start server
cd server
python main.py

# Start client
cd client
python main.py
```

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
```

---

## 📖 Documentation

- [Project Technical Proposal](docs/ProjectTechnicalProposal.md)
- [Developer Implementation Roadmap](docs/ProjectDeveloperImplementationRoadmap.md)
- [Code Requirements](docs/CodeRequirements.md)
- [Folder Structure Governance](docs/FolderStructureGovernance.md)

---

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to contribute to this project.

## 📞 Support

Need help? Check out our [Support Guide](SUPPORT.md) for various ways to get assistance:

- 🐛 [Report a Bug](https://github.com/YourOrg/Raspberry-Pi5-Federated/issues/new?template=bug_report.yml)
- ✨ [Request a Feature](https://github.com/YourOrg/Raspberry-Pi5-Federated/issues/new?template=feature_request.yml)
- ❓ [Ask a Question](https://github.com/YourOrg/Raspberry-Pi5-Federated/issues/new?template=question.yml)
- 💬 [Join Discussions](https://github.com/YourOrg/Raspberry-Pi5-Federated/discussions)

## 🔒 Security

Please review our [Security Policy](SECURITY.md) for information on reporting security vulnerabilities.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Raspberry Pi Foundation](https://www.raspberrypi.org/) for the amazing hardware
- [TensorFlow Federated](https://www.tensorflow.org/federated) team for the federated learning framework
- [Flower](https://flower.dev/) team for the federated learning platform
- All [contributors](https://github.com/YourOrg/Raspberry-Pi5-Federated/graphs/contributors) who help make this project better

---

## 🔗 Hardware Datasheets

- [Raspberry Pi 5 Product Brief](https://datasheets.raspberrypi.com/rpi5/raspberry-pi-5-product-brief.pdf)
- [Sense HAT Product Brief](https://datasheets.raspberrypi.com/sense-hat/sense-hat-product-brief.pdf)
- [ADXL345 Datasheet](https://www.analog.com/media/en/technical-documentation/data-sheets/adxl345.pdf)

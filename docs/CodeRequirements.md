# Projects Code Requirements  
Defines the mandatory code-level requirements...  
(Refer docs/CodeRequirements.md for full details)

# Projects Code Requirements

## 1. Overview  
Defines the mandatory code‐level requirements for the All-in-One IoT Edge project on Raspberry Pi, covering federated learning, TinyML inference, and MATLAB integration. Ensures consistency, maintainability, security, and ease of contribution.

---

## 2. Programming Languages & Runtimes  
- **Python 3.11** (minimum)  
  - All server & client logic must be compatible with CPython 3.11  
- **C / C++ (C++17)**  
  - For TensorFlow Lite Micro wrappers and microTVM runtime code  
- **MATLAB® (R2022b or newer)**  
  - Optional, for Engine API calls and Simulink code generation  
- **GNU Octave** (v7.x)  
  - Fallback for MATLAB scripts via Oct2Py  

---

## 3. Frameworks & Libraries  
- **Federated Learning**  
  - Flower ≥ 1.4 (​Apache 2.0) or TensorFlow Federated ≥ 0.34  
- **Core ML & TinyML**  
  - TensorFlow 2.12+ & Keras  
  - TensorFlow Lite (Python + Micro)  
  - Apache TVM ≥ 0.10 + microTVM  
- **MATLAB Integration**  
  - `matlabengine` Python package (MathWorks)  
  - VS Code MATLAB extension (MIT)  
  - Oct2Py ≥ 5.0 for Octave support  
- **Utilities**  
  - NumPy, Pandas, Matplotlib (BSD licenses)  
  - pybind11 ≥ 2.10 for C++/Python bridges  
  - CMake ≥ 3.20, Docker, Dev Containers  

---

## 4. Project Structure & Workspace  
- **Multi-root VS Code**  
  - `server/` – FL aggregator code  
  - `client/` – Pi-side sensing, ML, MATLAB calls  
  - `matlab/` – .m, .mlx, .slx, code-generation outputs  
- **Per-root Configuration**  
  - `.vscode/settings.json` specifying Python interpreter & linting rules  
  - `requirements.txt` (or `pyproject.toml`) per Python root  
  - `CMakeLists.txt` for building any native modules in `client/`  

---

## 5. Coding Standards & Style  
- **Python**  
  - Follow PEP 8; enforce via Black (auto-format) and Flake8 (lint)  
  - Docstrings in Google style or NumPy style for all public functions  
- **C/C++**  
  - Google C++ Style Guide conventions  
  - Enable `-Wall -Wextra -Werror` in compilation  
- **MATLAB**  
  - Use function headers and inline comments; adhere to MATLAB Code Analyzer suggestions  
- **Version Control**  
  - Git with signed commits (DCO)  
  - Feature branches named `feature/<short-description>`  

---

## 6. Dependency Management  
- **Pin exact versions** in `requirements.txt` or `environment.yml`  
- **Dependabot** configured for Python, Docker, and GitHub Actions  
- **Third-party licenses** documented in `THIRD_PARTY_LICENSES.md`  
- **No hard-coded credentials**; use environment variables or VS Code secrets  

---

## 7. Configuration & Secrets  
- **Config files**  
  - `config.yaml` or `JSON` for network endpoints, training hyperparameters, sensor sampling rates  
- **Secret management**  
  - Place tokens/certs outside repo (e.g. `.env`, GitHub Secrets)  
  - TLS certificates for Flower stored securely and loaded at runtime  

---

## 8. Logging & Monitoring  
- **Structured logging** (JSON format) via Python’s `logging` module  
- **Log levels**: DEBUG (development), INFO (runtime), WARN/ERROR (issues)  
- **Log retention**: rotate daily, keep 7 days by default  
- **Client metrics**: round durations, loss values, inference latency  

---

## 9. Testing & Validation  
- **Unit Tests** (pytest)  
  - ≥ 80% coverage for Python modules  
- **Integration Tests**  
  - In-process federated simulation (Flower) for 1–3 clients  
  - microTVM model compile + inference correctness  
  - MATLAB/Octave fallback smoke tests via Oct2Py  
- **Hardware-in-Loop (Manual)**  
  - Validate on physical Pi 5 + Sense HAT + ADXL345 before releases  
- **Test Data**  
  - Synthetic and/or anonymized real-world sample datasets under `tests/data/`  

---

## 10. Security & Compliance  
- **Secure Communications**  
  - Flower with TLS and token-based authentication  
- **Dependency Auditing**  
  - `pip-audit` or `safety` in CI to flag vulnerabilities  
- **Model Privacy**  
  - Integrate Differential Privacy &/or Secure Aggregation patterns  
- **Code Scanning**  
  - Bandit for Python, cppcheck for C/C++ in CI  

---

## 11. CI/CD Requirements  
- **GitHub Actions** pipelines:  
  1. **Lint & Format** stage (Black, Flake8, CMake lint)  
  2. **Unit & Integration Tests** stage (pytest, microTVM smoke)  
  3. **Performance Benchmarks** (autoencoder latency < 10 ms/sample; FL round < 5 s)  
  4. **Security Checks** (pip-audit, license scan)  
- **Artifacts**  
  - Test coverage reports, benchmark logs, Docker images (server & client)  

---

## 12. Performance Instrumentation  
- **Timing decorators** around:  
  - `client.fit()` (local training)  
  - `server.aggregate()` (FL aggregation)  
  - `autoencoder.invoke()` (inference)  
  - MATLAB engine calls (`eng.someFunction()`)  
- **Metrics export** to JSON or Prometheus endpoint for external dashboards  

---

## 13. MATLAB Integration Points  
- **Engine Startup**: only once per process (`matlab.engine.start_matlab()`)  
- **Script Execution**: use `eng.run('scriptname', nargout=0)` or function calls  
- **Data Exchange**: convert NumPy ↔ MATLAB arrays via `matlab.double()`  
- **Simulink**: headless runs via `eng.set_param('model','SimulationCommand','start')` or compiled Python package  

---

## 14. Build & Deployment  
- **Dev Containers**  
  - Dockerfile installs: Python, CMake, TF-F, TFLite, TVM, matlabengine (if available), octave-cli  
- **Client Packaging**  
  - Optional Docker image for Pi client; cross-compile native modules via QEMU in CI  
- **Deployment Scripts**  
  - Ansible playbook or shell scripts to provision Pi 5 nodes  

---

## 15. Documentation & Onboarding  
- **In-code docs**: docstrings, `README.md` per folder  
- **Project-wide docs**: `docs/` directory with Sphinx or MkDocs site  
- **Code Examples**: minimal snippets in README for FL server/client, autoencoder inference, MATLAB calls  
- **Getting Started Guide**: step-by-step in `docs/GETTING_STARTED.md`  

---

> _End of “Projects Code Requirements”_  

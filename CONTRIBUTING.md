# Contributing to Raspberry Pi 5 Federated Environmental Monitoring Network

Thank you for your interest in contributing to this project! This document provides guidelines and information for contributors.

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- Git
- VS Code (recommended)
- Docker (for Dev Containers)

### Development Setup
1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/Raspberry-Pi5-Federated.git
   cd Raspberry-Pi5-Federated
   ```
3. Open in VS Code and use Dev Container (recommended):
   - Open VS Code: `code IoT_Edge.code-workspace`
   - Command Palette: "Remote-Containers: Reopen in Container"

### Alternative Local Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.lock
# For server or client only:
# pip install -r server/requirements.lock
# pip install -r client/requirements.lock
pip install -r requirements-dev.txt
```

## 🏗️ Project Structure

- `server/` - Federated learning aggregator
- `client/` - Edge node code (sensing, ML, TinyML)
- `matlab/` - MATLAB/Simulink integration
- `docs/` - Documentation
- `tests/` - Test suites
- `scripts/` - Utility scripts

## 🔧 Development Workflow

### 1. Create a Feature Branch
```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes
- Follow the coding standards (see below)
- Add tests for new functionality
- Update documentation as needed

### 3. Run Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run linting
flake8 .
black --check .
```

### 4. Commit Changes
```bash
git add .
git commit -m "feat: add your feature description"
```

### 5. Push and Create PR
```bash
git push origin feature/your-feature-name
```
Then create a Pull Request on GitHub.

## 📝 Coding Standards

### Python
- Follow PEP 8
- Use Black for formatting
- Use Flake8 for linting
- Add type hints where appropriate
- Write docstrings for all public functions

### Code Formatting
```bash
# Format code
black .

# Sort imports
isort .

# Lint code
flake8 .
```

### Testing
- Write unit tests for all new functions
- Aim for >80% test coverage
- Use pytest for testing framework
- Place tests in `tests/` directory

## 🏷️ Commit Message Convention

Use conventional commits format:
- `feat:` - New features
- `fix:` - Bug fixes
- `docs:` - Documentation changes
- `test:` - Adding tests
- `refactor:` - Code refactoring
- `chore:` - Maintenance tasks

Example: `feat: add LSTM model for environmental forecasting`

## 🐛 Bug Reports

When reporting bugs, please include:
- Clear description of the issue
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, etc.)
- Relevant logs or error messages

## 💡 Feature Requests

For feature requests:
- Describe the feature and its benefits
- Provide use cases
- Consider implementation complexity
- Check if it aligns with project goals

## 📋 Pull Request Guidelines

### Before Submitting
- [ ] Tests pass locally
- [ ] Code is formatted (Black, isort)
- [ ] Linting passes (Flake8)
- [ ] Documentation updated
- [ ] CHANGELOG updated (if applicable)

### PR Description
- Clear title and description
- Link to related issues
- List changes made
- Include testing instructions

## 🔒 Security

- Never commit secrets or credentials
- Use environment variables for configuration
- Follow security best practices
- Report security issues privately

## 📚 Documentation

- Update relevant documentation for changes
- Use clear, concise language
- Include code examples where helpful
- Keep README.md up to date

## 🤝 Code of Conduct

This project follows a Code of Conduct. Please be respectful and inclusive in all interactions.

## 🆘 Getting Help

- Check existing issues and documentation
- Ask questions in GitHub Discussions
- Join our community channels (if available)

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to the Raspberry Pi 5 Federated Environmental Monitoring Network! 🎉

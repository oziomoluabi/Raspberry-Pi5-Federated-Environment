#!/usr/bin/env python3
"""
Setup script for Raspberry Pi 5 Federated Environmental Monitoring Network
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements, filtering out -r references
def read_requirements(filename):
    requirements = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('-r'):
                    requirements.append(line)
    except FileNotFoundError:
        pass
    return requirements

# Base requirements
base_requirements = [
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    "matplotlib>=3.7.0",
    "structlog>=23.1.0",
    "pyyaml>=6.0",
    "python-dotenv>=1.0.0",
]

setup(
    name="raspberry-pi5-federated",
    version="0.1.0",
    author="Raspberry Pi 5 Federated Team",
    author_email="team@example.com",
    description="Federated Environmental Monitoring Network with TinyML on Raspberry Pi 5",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/YourOrg/Raspberry-Pi5-Federated",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Hardware",
        "Topic :: System :: Networking",
    ],
    python_requires=">=3.11",
    install_requires=base_requirements,
    extras_require={
        "dev": read_requirements("requirements-dev.txt"),
        "server": read_requirements("server/requirements.txt"),
        "client": read_requirements("client/requirements.txt"),
    },
    entry_points={
        "console_scripts": [
            "federated-server=server.main:main",
            "federated-client=client.main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)

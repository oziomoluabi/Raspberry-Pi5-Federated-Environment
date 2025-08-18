FROM python:3.13.7-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    gcc \
    g++ \
    pkg-config \
    libssl-dev \
    libffi-dev \
    libblas-dev \
    liblapack-dev \
    gfortran \
    octave \
    octave-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Docker CLI (for building images)
RUN curl -fsSL https://get.docker.com | sh

# Set working directory
WORKDIR /workspace

# Copy requirements and install Python dependencies
COPY requirements.txt requirements-dev.txt ./
COPY server/requirements.txt ./server/
COPY client/requirements.txt ./client/

# Install Python packages
RUN pip install --no-cache-dir -r requirements-dev.txt

# Install TVM from source (for microTVM support)
RUN git clone --recursive https://github.com/apache/tvm.git /tmp/tvm && \
    cd /tmp/tvm && \
    mkdir build && cd build && \
    cmake -DUSE_LLVM=ON -DUSE_RPC=ON -DUSE_MICRO=ON .. && \
    make -j$(nproc) && \
    cd ../python && pip install -e . && \
    rm -rf /tmp/tvm

# Create non-root user
RUN useradd -m -s /bin/bash developer && \
    chown -R developer:developer /workspace

USER developer

# Set environment variables
ENV PYTHONPATH=/workspace
ENV TVM_HOME=/usr/local/lib/python3.11/site-packages/tvm

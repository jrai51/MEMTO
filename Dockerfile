# Start from the NVIDIA CUDA image with PyTorch-compatible CUDA version
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install essential packages and Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-dev python3-venv \
    build-essential git curl && \
    rm -rf /var/lib/apt/lists/*

# Safely create symlinks for Python and pip
RUN if [ ! -e /usr/bin/python ]; then ln -s /usr/bin/python3 /usr/bin/python; fi && \
    if [ ! -e /usr/bin/pip ]; then ln -s /usr/bin/pip3 /usr/bin/pip; fi

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip

# Copy your requirements.txt file to the container
COPY requirements.txt /workspace/requirements.txt

# Install Python packages from requirements.txt, including CUDA-compatible PyTorch
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117 && \
    pip install --no-cache-dir -r /workspace/requirements.txt

# Set the working directory
WORKDIR /workspace

# Default command to start an interactive shell; can be changed if running a specific script
CMD ["bash"]

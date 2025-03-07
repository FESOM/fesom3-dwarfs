FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/miniconda3/bin:${PATH}"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    git \
    build-essential \
    openmpi-bin \
    libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh \
    && bash miniconda.sh -b -p /root/miniconda3 \
    && rm miniconda.sh

# Create conda environment and install dependencies
RUN conda create -n fesom python=3.9 -y \
    && conda install -n fesom -y \
    numpy \
    mpi4py \
    && conda clean -afy

SHELL ["conda", "run", "-n", "fesom", "/bin/bash", "-c"]

# Install JAX with CUDA support
RUN pip install --no-cache-dir \
    "jax[cuda12_pip]" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Set working directory
WORKDIR /app

# Copy the source code
COPY . /app/

# Set default command to activate conda environment
ENTRYPOINT ["conda", "run", "-n", "fesom"]
CMD ["python"]

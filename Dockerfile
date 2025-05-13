FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Install system dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    ffmpeg \
    python3.10 \
    python3.10-venv \
    python3-pip \
 && rm -rf /var/lib/apt/lists/*

# Set python3.10 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Set working directory
WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .

# Install Python dependencies
# Install torch compatible with CUDA 12.1 first
RUN pip3 install --no-cache-dir torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0+cu121 --index-url https://download.pytorch.org/whl/cu121
# Install remaining requirements, filter out torch, torchvision and ffmpeg which are handled separately
RUN grep -vE '^torch==|^torchvision|^ffmpeg==' requirements.txt > requirements_filtered.txt && \
    pip3 install --no-cache-dir -r requirements_filtered.txt && \
    rm requirements_filtered.txt

# Set Hugging Face cache directory to a location inside the container
ENV TRANSFORMERS_CACHE=/app/hf_cache
ENV HF_HOME=/app/hf_cache

# Pre-download and cache transformers models
RUN mkdir -p /app/hf_cache && \
    python3 -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('openai/clip-vit-large-patch14'); AutoModel.from_pretrained('openai/clip-vit-large-patch14')"

# pre-trained model download
# RUN pip3 install gdown && gdown https://drive.google.com/uc?id=17-ZqLJ0gNJGqlO0Mu0Hyoi31fLp0dKWY

# Copy the rest of the application code
COPY . .

# Create directory for pretrained models (user needs to mount/copy models here)
# RUN mkdir -p pretrained_models/upscale_a_video

# Expose any necessary ports (if applicable, none specified)
# ENV PYTHONUNBUFFERED=1

# Define the entry point for running inference
# ENTRYPOINT ["python3", "inference_upscale_a_video.py"]
ENTRYPOINT ["/bin/bash"]

# Default command arguments (optional, user can override)
# CMD ["-i", "./inputs/aigc_1.mp4", "-o", "./results"] 
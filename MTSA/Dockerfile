FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir flash-attn --no-build-isolation

# Copy project
COPY . .

# Set env vars
ENV PYTHONPATH=/app
ENV HF_HOME=/app/cache

# Default command
CMD ["bash", "script/slurm/run_rlvr_defence.sh"]

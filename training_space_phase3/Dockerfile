FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DEFAULT_TIMEOUT=300

RUN apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3.11-venv python3-pip \
    build-essential git curl \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && python3 -m pip install --upgrade pip setuptools wheel

# Pinned stable torch 2.5.1 + cu121 — matches container CUDA 12.1.1 exactly
RUN pip install \
    torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Pinned stable training stack — no unsloth, no xformers
# trl 0.15+ has GRPOTrainer; transformers 4.48+ for compat
RUN pip install \
    "transformers==4.48.3" \
    "accelerate==1.3.0" \
    "peft==0.14.0" \
    "trl==0.15.2" \
    "bitsandbytes==0.45.0" \
    "datasets==3.2.0" \
    "huggingface_hub>=0.27.0" \
    wandb requests pydantic fastapi uvicorn matplotlib

WORKDIR /app

COPY dead_internet_detective/ ./dead_internet_detective/
COPY client/ ./client/
COPY train.py ./train.py
COPY app.py ./app.py

EXPOSE 7860

CMD ["python3", "app.py"]

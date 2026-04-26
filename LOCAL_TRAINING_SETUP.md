# Local Training Setup — RTX 4070 Laptop (Windows + WSL2)

## Step 1 — Enable WSL2 (run in Windows PowerShell as Administrator)
```powershell
wsl --install
wsl --set-default-version 2
# Reboot if prompted. After reboot:
wsl --install -d Ubuntu-22.04
```

## Step 2 — Verify GPU is visible inside WSL2
```bash
# Open Ubuntu terminal (WSL2), run:
nvidia-smi
# You must see your RTX 4070 listed.
# If you do NOT see it: your NVIDIA driver version supports WSL2
# (driver 32.0.15.9621 is well above the minimum 525.x required)
# Try: wsl --update then reboot
```

## Step 3 — Install Python + CUDA toolkit inside WSL2
```bash
sudo apt update && sudo apt install -y python3.11 python3.11-venv python3-pip git
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
# Must print: True and RTX 4070
```

## Step 4 — Clone repo and install dependencies inside WSL2
```bash
git clone https://github.com/aj7075/dead-internet-detective.git
cd dead-internet-detective
pip3 install -r requirements.txt
pip3 install "unsloth[cu121] @ git+https://github.com/unslothai/unsloth.git"
pip3 install trl wandb matplotlib datasets
```

## Step 5 — Verify Unsloth sees the GPU
```bash
python3 -c "from unsloth import FastLanguageModel; print('Unsloth OK')"
```

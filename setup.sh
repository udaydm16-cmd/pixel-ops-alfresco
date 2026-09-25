#!/bin/bash
# Pixel-Ops - Lighthouse dataset - environment setup (Linux/Ubuntu)
# Installs COLMAP as a command-line/backend tool (NOT a GUI reconstruction app)
# plus the Python libraries used by our own pipeline scripts.
set -e

echo "== Updating package lists =="
sudo apt-get update

echo "== Installing COLMAP (CLI backend) =="
sudo apt-get install -y colmap

# If your Ubuntu's apt version of COLMAP is very old, build from source instead:
#   https://colmap.github.io/install.html
# but for a hackathon timeline, apt's version is fine.

echo "== Checking for NVIDIA GPU (optional, speeds up SIFT + stereo a lot) =="
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected. COLMAP will use GPU acceleration where flagged."
    GPU_FLAG=1
else
    echo "No GPU detected. Pipeline will run on CPU (slower, but fine for ~200 images overnight)."
    GPU_FLAG=0
fi
echo "$GPU_FLAG" > /tmp/pixelops_gpu_flag

echo "== Installing Python dependencies =="
pip install --break-system-packages open3d opencv-python numpy matplotlib

echo "== Setup complete =="
echo "Next: copy your 200 Lighthouse images into project/images/ and run scripts/run_colmap_pipeline.sh"

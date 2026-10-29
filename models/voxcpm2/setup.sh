#!/usr/bin/env bash
# Installs VoxCPM2 into /kaggle/working/venv and downloads the weights (about 5 GB) into the Hugging Face cache.
set -e
cd /kaggle/working
pip install -q -U uv
if [ ! -d VoxCPM ]; then git clone -q --depth 1 https://github.com/OpenBMB/VoxCPM.git; fi
if [ ! -x venv/bin/python ]; then uv venv --python 3.11 venv; fi
uv pip install --python venv/bin/python -e VoxCPM fastapi uvicorn python-multipart psutil soundfile pyngrok
venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('${VOXCPM_MODEL:-openbmb/VoxCPM2}')"
echo "VoxCPM2 is installed."

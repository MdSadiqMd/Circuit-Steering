# shellcheck shell=bash
# Source before every command on puget. Everything lives on the /dev/shm RAM disk because the root
# disk is full; nothing is written under $HOME. cs_puget.pth + cs_puget_setup.py go into the venv's
# site-packages (CUDA shims and the 30 GB GPU cap, read by cs_puget_setup.py).
export CS=/dev/shm/cs
export PATH=/home/user/.local/bin:$PATH
export UV_CACHE_DIR=$CS/uvcache UV_PYTHON_INSTALL_DIR=$CS/py UV_LINK_MODE=copy
export TMPDIR=$CS/tmp HF_HOME=$CS/hf KE_CACHE_DIR=$CS/ke_cov HF_HUB_OFFLINE=1
export XDG_CACHE_HOME=$CS/xdg MPLCONFIGDIR=$CS/xdg/mpl TRITON_CACHE_DIR=$CS/xdg/triton
export CUBLAS_WORKSPACE_CONFIG=:4096:8 CS_PUGET=1 CS_GPU_CAP_GB=30 TOKENIZERS_PARALLELISM=false
export CIRCUITKIT_TL_MEMORY_PATCH=1

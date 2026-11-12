"""Puget-only process setup, loaded by `cs_puget.pth` in the puget venv's site-packages (a
`sitecustomize` is shadowed by the system Python's own) and active only with
CS_PUGET=1 (scripts/puget/env.sh). The pipeline was written for MPS; on CUDA three things differ:
  - `torch.mps.empty_cache()` raises without an MPS backend and several stages call it
    unconditionally, so it becomes a no-op;
  - the 30 GB budget is enforced on the GPU with the caching allocator's per-process fraction
    (MemWatch only sees host RSS on CUDA);
  - fp32 matmuls stay true fp32 (no TF32) and cuBLAS runs with a fixed workspace
    (CUBLAS_WORKSPACE_CONFIG), with deterministic algorithms requested and a warning, not an error,
    where an op has none."""

import os

if os.environ.get("CS_PUGET") == "1":
    import torch

    if not torch.backends.mps.is_available():
        torch.mps.empty_cache = lambda: None
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    cap = float(os.environ.get("CS_GPU_CAP_GB", "0"))
    if cap and torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, cap * 2**30 / total))

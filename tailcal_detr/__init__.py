import os

# If a user enables deterministic algorithms on CUDA, some ops require a cuBLAS
# workspace config env var. Setting a default here prevents runtime crashes
# when `repro.deterministic: true` and `device: auto` selects CUDA.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

__all__ = [
    "__version__",
]

__version__ = "0.1.0"

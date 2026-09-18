"""Isolate tests that deliberately enter a production precision runtime."""
import pytest
import torch


@pytest.fixture(autouse=True)
def restore_torch_runtime():
    threads = torch.get_num_threads()
    precision = torch.get_float32_matmul_precision()
    cuda_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    rng_state = torch.get_rng_state()
    try:
        yield
    finally:
        torch.set_num_threads(threads)
        torch.set_float32_matmul_precision(precision)
        torch.backends.cuda.matmul.allow_tf32 = cuda_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.set_rng_state(rng_state)

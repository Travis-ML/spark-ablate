"""Model/tokenizer loading with DGX Spark-friendly defaults.

The Spark's 128 GB unified memory means big models load fine without
offloading tricks: bf16 on a single cuda device is the default. device_map
('auto') is only used when explicitly requested, since it requires
accelerate and buys nothing on a single-GPU box.

Auto device selection falls back to Apple MPS, then CPU, so the same
configs run unmodified on a Mac for development.
"""

from __future__ import annotations

import warnings

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def pick_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _check_cuda_arch() -> None:
    """Warn if the torch build lacks kernels for this GPU's compute capability.

    The DGX Spark's GB10 is sm_121; generic wheels predating CUDA 12.8/13
    don't include it and either PTX-JIT slowly or fail with 'no kernel
    image'. Catch that at load time instead of mid-sweep.
    """
    major, minor = torch.cuda.get_device_capability()
    arch = f"sm_{major}{minor}"
    arch_list = torch.cuda.get_arch_list()
    if arch_list and arch not in arch_list:
        warnings.warn(
            f"GPU compute capability {arch} is not in this torch build's "
            f"arch list {arch_list}. On a DGX Spark (GB10, sm_121) use "
            "NVIDIA's PyTorch NGC container or an aarch64 CUDA wheel built "
            "for Blackwell, or kernels may PTX-JIT slowly or fail to launch.",
            stacklevel=3,
        )


def load_model(name: str, dtype: str = "bfloat16", device: str = "auto",
               device_map: str | None = None, trust_remote_code: bool = False):
    """Returns (model, tokenizer) in eval mode with grads disabled."""
    if dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {sorted(DTYPES)}, got {dtype!r}")
    torch_dtype = DTYPES[dtype]
    resolved = pick_device(device)
    if resolved == "cpu" and torch_dtype != torch.float32:
        torch_dtype = torch.float32  # half precision on CPU is slow/flaky
    elif resolved == "mps" and torch_dtype == torch.bfloat16:
        torch_dtype = torch.float16  # bf16 on MPS is incomplete; fp16 is the safe half type
    elif resolved.startswith("cuda") and torch.cuda.is_available():
        _check_cuda_arch()

    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: dict = {"trust_remote_code": trust_remote_code}
    if device_map:
        kwargs["device_map"] = device_map
    try:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch_dtype, **kwargs)
    except TypeError:  # transformers < 4.56 uses torch_dtype=
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch_dtype, **kwargs)

    if not device_map:
        model.to(resolved)
    model.eval()
    model.requires_grad_(False)
    return model, tokenizer

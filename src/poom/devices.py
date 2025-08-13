import torch

def get_device(preference: str = "auto") -> torch.device:
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    if preference == "cpu":
        return torch.device("cpu")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_dtype(name: str):
    name = name.lower()
    if name in ("float32", "fp32"):
        return torch.float32
    if name in ("float64", "fp64", "double"):
        return torch.float64
    raise ValueError(f"Unsupported dtype: {name}")

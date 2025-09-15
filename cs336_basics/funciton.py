import torch
from jaxtyping import Float, Int
from einops import reduce, einsum
import math
from collections.abc import Iterable
from numpy import ndarray
import numpy as np
import os
import typing


def softmax(
    in_features: Float[torch.Tensor, " ..."],
    dimension: int
) -> Float[torch.Tensor, " ..."]:
    max_values = in_features.max(dim=dimension, keepdim=True).values
    in_features = in_features - max_values + 1
    
    e_x = in_features.exp()
    sum_e_x = e_x.sum(dim=dimension, keepdim=True)
    
    return e_x / sum_e_x
    
    

def log_softmax(
    in_features: Float[torch.Tensor, " ..."],
    dimension: int
) -> Float[torch.Tensor, " ..."]:
    max_values = in_features.max(dim=dimension, keepdim=True).values
    in_features = in_features - max_values + 1
    
    e_x = in_features.exp()
    log_e_x = e_x.sum(dimension, keepdim=True).log()
    
    return in_features - log_e_x


def learning_rate_schedule(
    t: int,
    lr_max: float,
    lr_min: float,
    warmup_t: int,
    cos_cycle_t: int,
) -> float:
    if t < warmup_t:
        return t / warmup_t * lr_max
    
    if t <= cos_cycle_t:
        return lr_min + 0.5 * (1 + math.cos((t - warmup_t) / (cos_cycle_t - warmup_t) * math.pi)) * (lr_max - lr_min)
    
    return lr_min
    
def gradient_clipping(
    params: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
    eps: float = 1e-6
) -> None:
    norm = 0
    for p in params:
        if p.grad is None:
            continue
        grad = p.grad.data
        norm += torch.sum(grad ** 2)
    norm **= 0.5
    
    if norm > max_l2_norm:
        for p in params:
            if p.grad is None:
                continue
            p.grad.data *= max_l2_norm / (norm + eps)
    

def data_loading(
    x: ndarray,
    batch_size: int,
    context_length: int, 
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = x.shape[0] - context_length - 1
    starts = np.random.randint(0, max_start + 1, size=batch_size)
    
    inputs = np.stack([x[s:s + context_length] for s in starts]).astype(np.int64)
    targets = np.stack([x[s + 1:s + context_length + 1] for s in starts]).astype(np.int64)
    
    inputs_tensor: Int[torch.Tensor, " batch_size context_legnth"] = torch.tensor(inputs).to(device)
    targets_tensor: Int[torch.Tensor, " batch_size context_legnth"] = torch.tensor(targets).to(device)
    
    return inputs_tensor, targets_tensor



def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    if isinstance(out, (str, os.PathLike)):
        parent = os.path.dirname(str(out))
        if parent:
            os.makedirs(parent, exist_ok=True)
    torch.save(checkpoint, out)

def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]
    


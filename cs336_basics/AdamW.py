from typing import Any, Dict
import torch
from torch.optim.optimizer import ParamsT
from typing import Optional
from collections.abc import Callable
from collections import defaultdict

class AdamW(torch.optim.Optimizer):
    def __init__(
        self, 
        params: ParamsT,
        lr: float, 
        betas: tuple[float, float], 
        eps: float,
        weight_decay: float,
    ) -> None:
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay
        }
        super().__init__(params, defaults)
        
    def step( # type: ignore
        self,
        closure: Optional[Callable] = None
    ) -> Optional[float]:
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            betas = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            
            for p in group["params"]:
                p: torch.Tensor
                state:defaultdict = self.state[p]
                
                if p.grad is None:
                    continue
                grad = p.grad.data
                
                m = state.get("m", torch.zeros_like(p))
                v = state.get("v", torch.zeros_like(p))
                m = m * betas[0] + (1 - betas[0]) * grad
                v = v * betas[1] + (1 - betas[1]) * (grad ** 2)
                state["m"] = m
                state["v"] = v
                
                t = state.get("t", 1)
                lr_t = lr * (1 - betas[1] ** t) ** 0.5 / (1 - betas[0] ** t)
                p.data = p.data - lr_t * m / (v ** 0.5 + eps)
                p.data = p.data - lr * weight_decay * p.data
                
                state["t"] = t + 1
                
        return loss
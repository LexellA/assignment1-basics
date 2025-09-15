from collections.abc import Callable, Iterable
from typing import Dict, Optional
import torch
import math

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr) -> None:
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)
        
    def step(self, closure: Optional[Callable] = None): # type: ignore
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1
        
        return loss
    
    
weights = torch.nn.Parameter(5 * torch.rand((10, 10)))
opt = SGD([weights], lr=1)
for t in range(1000):
    opt.zero_grad()
    loss = (weights ** 2).mean()
    print(loss.cpu().item())
    loss.backward()
    opt.step()
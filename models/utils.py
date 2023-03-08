import torch
import torch.nn as nn


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class EMA:
    def __init__(self):
        super().__init__()

    def update_average(self, old, new, beta):
        if old is None:
            return new
        return old * beta + (1 - beta) * new

    @torch.no_grad()
    def update_moving_average(self, ma_model, current_model, beta):
        for current_params, ma_params in zip(
            current_model.parameters(), ma_model.parameters()
        ):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight, beta)


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

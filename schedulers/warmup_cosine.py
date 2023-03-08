import numpy as np
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, num_epochs, iter_per_epoch, min_lr=1e-5):
        self.base_lr = optimizer.param_groups[0]["lr"]
        self.warmup_iter = warmup_epochs * iter_per_epoch
        self.total_iter = num_epochs * iter_per_epoch
        self.optimizer = optimizer
        self.iter = 0
        self.current_lr = 0
        self.min_lr = min_lr

        self.init_lr()  # So that at first step we have the correct step size

    def get_lr(self, base_lr):
        # Linear warmup
        if self.iter < self.warmup_iter:
            return base_lr * self.iter / self.warmup_iter
        # Cosine decay
        else:
            decay_iter = self.total_iter - self.warmup_iter
            return self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * (self.iter - self.warmup_iter) / decay_iter))

    def update_param_groups(self):
        self.optimizer.param_groups[0]["lr"] = self.get_lr(self.base_lr)

    def step(self):
        self.update_param_groups()
        self.iter += 1

    def init_lr(self):
        self.update_param_groups()


class WarmupCosineMomentumScheduler:
    def __init__(
        self,
        base_momentum,
        warmup_epochs,
        num_epochs,
        iter_per_epoch,
        cosine_decay=True,
    ):
        self.base_val = 1.0 - base_momentum
        self.warmup_iter = warmup_epochs * iter_per_epoch
        self.total_iter = num_epochs * iter_per_epoch
        self.current_lr = 0
        self.cosine_decay = cosine_decay

    def get_lr(self, step):
        if step < self.warmup_iter:
            return 1.0 - self.base_val * step / self.warmup_iter
        elif not self.cosine_decay:
            return 1.0 - self.base_val
        else:
            decay_iter = self.total_iter - self.warmup_iter
            return 1.0 - 0.5 * self.base_val * (
                1 + np.cos(np.pi * (step - self.warmup_iter) / decay_iter)
            )

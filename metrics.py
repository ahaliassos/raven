import numpy as np
import torch
from torchmetrics import Metric


def get_wer(s, ref):
    return get_er(s.split(), ref.split())


def get_cer(s, ref):
    return get_er(list(s), list(ref))


def get_er(s, ref):
    """
    FROM wikipedia levenshtein distance
    s: list of words/char in sentence to measure
    ref: list of words/char in reference
    """

    costs = np.zeros((len(s) + 1, len(ref) + 1))
    for i in range(len(s) + 1):
        costs[i, 0] = i
    for j in range(len(ref) + 1):
        costs[0, j] = j

    for j in range(1, len(ref) + 1):
        for i in range(1, len(s) + 1):
            cost = None
            if s[i - 1] == ref[j - 1]:
                cost = 0
            else:
                cost = 1
            costs[i, j] = min(
                costs[i - 1, j] + 1, costs[i, j - 1] + 1, costs[i - 1, j - 1] + cost
            )

    return costs[-1, -1] / len(ref)


class WER(Metric):
    def __init__(self, dist_sync_on_step=False):
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, target):
        n = len(target.split())
        self.error += torch.tensor(get_wer(preds, target) * n)
        self.total += torch.tensor(n)

    def compute(self):
        # compute final result
        return self.error.float() / self.total

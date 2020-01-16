""" Operations """
import torch
import torch.nn as nn
import genotypes as gt


OPS = {
    # "none": lambda C: Zero(),
    # "skip_connect": lambda C: Identity(),
    "relu": lambda C: ReLu(),
    "linear_to_10": lambda C: Linear(C, 10),
}


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.l = nn.Linear(in_features, out_features, bias)

    def forward(self, x):
        out = self.l(x)
        return out


class ReLu(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.l = torch.nn.ReLU(inplace)

    def forward(self, x):
        return self.l(x)


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * 0.0


class MixedOp(nn.Module):
    """ Mixed operation """

    def __init__(self, C=10):
        super().__init__()
        self._ops = (
            nn.ModuleList()
        )  # select 2 ops randomly for each couple of nodes at the start of the training
        for primitive in gt.PRIMITIVES:
            op = OPS[primitive](C)
            self._ops.append(op)

    def forward(self, x, weights):
        """
        Args:
            x: input
            weights: weight for each operation
        """
        # for w, op in zip(weights, self._ops):
        #     print("w: ", w.size())
        #     print("op(x): ", op(x).size())
        return sum(
            w * op(x) for w, op in zip(weights, self._ops)
        )  # only sum for the 2 ops selected


""" Operations """
import torch
import torch.nn as nn
import genotypes as gt


OPS = {
    # "none": lambda C: Zero(),
    "identity": lambda in_f: Identity(in_f, 20),
    "ReLu": lambda in_f: ReLu(in_f, 20),
    "linear_10": lambda C: Linear(C, 10, (0, 10)),
    "linear_12": lambda C: Linear(C, 12, (0, 8)),
    "linear_14": lambda C: Linear(C, 14, (0, 6)),
    "linear_16": lambda C: Linear(C, 16, (0, 4)),
    "linear_18": lambda C: Linear(C, 18, (0, 2)),
    "linear_20": lambda C: Linear(C, 20, (0, 0)),
}


class Linear(nn.Module):
    def __init__(self, in_features, out_features, zero_padding=(0, 0), bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features + sum(zero_padding)
        self.l = nn.Linear(in_features, out_features, bias)
        self.p = nn.ConstantPad1d(zero_padding, 0)

    def forward(self, x):
        out = self.l(x)
        out = self.p(out)
        return out


class ReLu(nn.Module):
    def __init__(self, in_features, out_features, inplace=False):
        super().__init__()
        self.in_features = None
        self.out_features = None
        self.relu = torch.nn.ReLU(inplace)
        self.p = nn.ConstantPad1d((0, out_features - in_features), 0)

    def forward(self, x):
        return self.p(self.relu(x))


class Identity(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.p = nn.ConstantPad1d((0, out_features - in_features), 0)

    def forward(self, x):
        return self.p(x)


class Zero(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * 0.0


class MixedOp(nn.Module):
    """ Mixed operation """

    def __init__(self, in_features):
        super().__init__()
        self._ops = (
            nn.ModuleList()
        )  # select 2 ops randomly for each couple of nodes at the start of the training
        self.out_features = None
        for primitive in gt.PRIMITIVES:
            op = OPS[primitive](in_features)
            if op.out_features is not None:
                self.out_features = (
                    op.out_features
                    if self.out_features is None
                    else max(op.out_features, self.out_features)
                )
            self._ops.append(op)

    def forward(self, x, weights):
        """
        Args:
            x: input
            weights: weight for each operation
        """
        # res = None
        # for w, op in zip(weights, self._ops):
        #     print(w, op, op(x).size())
        #     if res is None:
        #         res = w * op(x)
        #     else:
        #         res += w * op(x)
        # return res
        return sum(
            w * op(x) for w, op in zip(weights, self._ops)
        )  # only sum for the 2 ops selected


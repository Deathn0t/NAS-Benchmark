""" CNN cell for architecture search """
import torch
import torch.nn as nn
from models import ops


class SearchCell(nn.Module):
    """ Cell for search
    Each edge is mixed and continuous relaxed.
    """

    def __init__(self, in_features, n_nodes):
        """
        Args:
            n_nodes: # of intermediate n_nodes
        """
        super().__init__()
        self.n_nodes = n_nodes
        self.in_features = in_features
        self.out_features = None

        # generate dag
        self.dag = nn.ModuleList()
        for i in range(self.n_nodes):
            self.dag.append(nn.ModuleList())
            op = ops.MixedOp(in_features)
            in_features = op.out_features
            self.dag[-1].append(op)
        self.out_features = in_features

        print(f"new cell(in={self.in_features}, out={self.out_features})")

    def forward(self, x, w_dag):
        # print("len zip: ", len(list(zip(self.dag, w_dag))))
        for ops, w_list in zip(self.dag, w_dag):
            x = sum(ops[i](s, w) for i, (s, w) in enumerate(zip([x], w_list)))
        return x

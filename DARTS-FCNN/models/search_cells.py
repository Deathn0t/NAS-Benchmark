""" CNN cell for architecture search """
import torch
import torch.nn as nn
from models import ops


class SearchCell(nn.Module):
    """ Cell for search
    Each edge is mixed and continuous relaxed.
    """

    def __init__(self, n_nodes):
        """
        Args:
            n_nodes: # of intermediate n_nodes
        """
        super().__init__()
        self.n_nodes = n_nodes

        # If previous cell is reduction cell, current input size does not match with
        # output size of cell[k-2]. So the output[k-2] should be reduced by preprocessing.

        # generate dag
        self.dag = nn.ModuleList()
        for i in range(self.n_nodes):
            # for j in range(1 + i):  # include 1 input nodes
            # reduction should be used only for input node
            self.dag.append(nn.ModuleList())
            op = ops.MixedOp()
            print("new mixop")
            self.dag[-1].append(op)

    def forward(self, x, w_dag):
        # print("len zip: ", len(list(zip(self.dag, w_dag))))
        for ops, w_list in zip(self.dag, w_dag):
            x = sum(ops[i](s, w) for i, (s, w) in enumerate(zip([x], w_list)))
            # x = op(x, w)

        return x

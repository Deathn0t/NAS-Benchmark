""" CNN for architecture search """
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.search_cells import SearchCell
import genotypes as gt
from torch.nn.parallel._functions import Broadcast
import logging


def broadcast_list(l, device_ids):
    """ Broadcasting list """
    l_copies = Broadcast.apply(device_ids, *l)
    l_copies = [l_copies[i : i + len(l)] for i in range(0, len(l_copies), len(l))]

    return l_copies


class SearchFCNN(nn.Module):
    """ Search CNN model """

    def __init__(self, in_dim, out_dim, n_layers, n_nodes=4):
        """
        Args:
            n_outputs: # of classes
            n_layers: # of layers
            n_nodes: # of intermediate nodes in Cell
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_layers = n_layers

        self.cells = nn.ModuleList()
        for i in range(n_layers):
            cell = SearchCell(in_dim, n_nodes)
            in_dim = cell.out_features
            self.cells.append(cell)
        self.linear = nn.Linear(in_dim, self.out_dim)

    def forward(self, x, weights_normal):

        for cell in self.cells:
            x = cell(x, weights_normal)

        logits = self.linear(x)
        return logits


class SearchFCNNController(nn.Module):
    """ SearchCNN controller supporting multi-gpu """

    def __init__(self, in_dim, out_dim, n_layers, criterion, n_nodes=4, device_ids=None):
        super().__init__()
        self.n_nodes = n_nodes
        self.criterion = criterion
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        self.device_ids = device_ids

        # initialize architect parameters: alphas
        n_ops = len(gt.PRIMITIVES)

        self.alpha_normal = nn.ParameterList()

        for i in range(n_nodes):
            # self.alpha_normal.append(nn.Parameter(1e-3 * torch.randn(i + 2, n_ops)))
            self.alpha_normal.append(nn.Parameter(1e-3 * torch.randn(1, n_ops)))

        print("len alpha: ", len(self.alpha_normal))

        # setup alphas list
        self._alphas = []
        for n, p in self.named_parameters():
            if "alpha" in n:
                self._alphas.append((n, p))

        self.net = SearchFCNN(in_dim, out_dim, n_layers, n_nodes)

    def forward(self, x):
        weights_normal = [F.softmax(alpha, dim=-1) for alpha in self.alpha_normal]

        if len(self.device_ids) == 1:
            return self.net(x, weights_normal)

        # scatter x
        xs = nn.parallel.scatter(x, self.device_ids)
        # broadcast weights
        wnormal_copies = broadcast_list(weights_normal, self.device_ids)

        # replicate modules
        replicas = nn.parallel.replicate(self.net, self.device_ids)
        outputs = nn.parallel.parallel_apply(
            replicas, list(zip(xs, wnormal_copies)), devices=self.device_ids
        )
        return nn.parallel.gather(outputs, self.device_ids[0])

    def loss(self, X, y):
        # print("X: ", X.size(), ", dtype: ", X.type())
        # print("y: ", y.size(), ", dtype: ", y.type())
        logits = self.forward(X)
        return self.criterion(logits, y)

    def print_alphas(self, logger):
        # remove formats
        org_formatters = []
        for handler in logger.handlers:
            org_formatters.append(handler.formatter)
            handler.setFormatter(logging.Formatter("%(message)s"))

        logger.info("####### ALPHA #######")
        logger.info("# Alpha - normal")
        for alpha in self.alpha_normal:
            logger.info(F.softmax(alpha, dim=-1))

        logger.info("#####################")

        # restore formats
        for handler, formatter in zip(logger.handlers, org_formatters):
            handler.setFormatter(formatter)

    def genotype(self):
        gene_normal = gt.parse(self.alpha_normal, k=1)
        concat = range(1, 1 + self.n_nodes)  # concat all intermediate nodes

        return gt.Genotype(normal=gene_normal, normal_concat=concat)

    def weights(self):
        return self.net.parameters()

    def named_weights(self):
        return self.net.named_parameters()

    def alphas(self):
        for n, p in self._alphas:
            yield p

    def named_alphas(self):
        for n, p in self._alphas:
            yield n, p

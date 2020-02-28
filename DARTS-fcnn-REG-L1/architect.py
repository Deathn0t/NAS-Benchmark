""" Architect controls architecture of cell by computing gradients of alphas """
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class Architect:
    """ Compute gradients of alphas """

    def __init__(self, net, w_momentum, w_weight_decay, allow_unused=False):
        """
        Args:
            net
            w_momentum: weights momentum
        """
        self.net = net
        self.v_net = copy.deepcopy(net)
        self.w_momentum = w_momentum
        self.w_weight_decay = w_weight_decay
        self.allow_unused = allow_unused
        self.old_alpha = copy.deepcopy(self.v_net.alpha_normal)
        # for alpha in self.v_net.alpha_normal:
        #     self.old_alpha.append(alpha.clone())
        # self.reg_crit = lambda (x, y): F.kl_div(torch.log(x), target=y, reduction="sum")

    def reg_crit(self, x, y):
        return F.kl_div(torch.log(x), target=y, reduction="sum")

    def virtual_step(self, trn_X, trn_y, xi, w_optim):
        """
        Compute unrolled weight w' (virtual step)

        Step process:
        1) forward
        2) calc loss
        3) compute gradient (by backprop)
        4) update gradient

        Args:
            xi: learning rate for virtual gradient step (same as weights lr)
            w_optim: weights optimizer
        """
        # forward & calc loss
        loss = self.net.loss(trn_X, trn_y)  # L_trn(w)

        # compute gradient
        gradients = torch.autograd.grad(
            loss, self.net.weights(), allow_unused=self.allow_unused
        )

        # do virtual step (update gradient)
        # below operations do not need gradient tracking
        with torch.no_grad():
            # dict key is not the value, but the pointer. So original network weight have to
            # be iterated also.
            for w, vw, g in zip(self.net.weights(), self.v_net.weights(), gradients):
                m = w_optim.state[w].get("momentum_buffer", 0.0) * self.w_momentum
                vw.copy_(w - xi * (m + g + self.w_weight_decay * w))

            # synchronize alphas
            for a, va in zip(self.net.alphas(), self.v_net.alphas()):
                va.copy_(a)

    def unrolled_backward(self, trn_X, trn_y, val_X, val_y, xi, w_optim, reg_crit):
        """ Compute unrolled loss and backward its gradients
        Args:
            xi: learning rate for virtual gradient step (same as net lr)
            w_optim: weights optimizer - for virtual step
        """
        # do virtual step (calc w`)
        self.virtual_step(trn_X, trn_y, xi, w_optim)

        # calc unrolled loss
        loss = self.v_net.loss(val_X, val_y)  # L_val(w`)
        reg_loss = 0
        for old_a, cur_a in zip(self.old_alpha, self.v_net.alpha_normal):
            reg_loss += self.reg_crit(cur_a, old_a)
        reg_factor = -0.5
        loss += reg_factor * reg_loss

        # compute gradient
        v_alphas = tuple(self.v_net.alphas())
        v_weights = tuple(self.v_net.weights())
        v_grads = torch.autograd.grad(
            loss, v_alphas + v_weights, allow_unused=self.allow_unused
        )
        dalpha = v_grads[: len(v_alphas)]
        dw = v_grads[len(v_alphas) :]

        hessian = self.compute_hessian(dw, trn_X, trn_y)

        # update final gradient = dalpha - xi*hessian
        with torch.no_grad():
            for alpha, da, h in zip(self.net.alphas(), dalpha, hessian):
                alpha.grad = da - xi * h

    def compute_hessian(self, dw, trn_X, trn_y):
        """
        dw = dw` { L_val(w`, alpha) }
        w+ = w + eps * dw
        w- = w - eps * dw
        hessian = (dalpha { L_trn(w+, alpha) } - dalpha { L_trn(w-, alpha) }) / (2*eps)
        eps = 0.01 / ||dw||
        """
        norm = torch.cat([w.view(-1) for w in dw]).norm()
        eps = 0.01 / norm

        # w+ = w + eps*dw`
        with torch.no_grad():
            for p, d in zip(self.net.weights(), dw):
                p += eps * d
        loss = self.net.loss(trn_X, trn_y)
        dalpha_pos = torch.autograd.grad(
            loss, self.net.alphas(), allow_unused=self.allow_unused
        )  # dalpha { L_trn(w+) }

        # w- = w - eps*dw`
        with torch.no_grad():
            for p, d in zip(self.net.weights(), dw):
                p -= 2.0 * eps * d
        loss = self.net.loss(trn_X, trn_y)
        dalpha_neg = torch.autograd.grad(
            loss, self.net.alphas(), allow_unused=self.allow_unused
        )  # dalpha { L_trn(w-) }

        # recover w
        with torch.no_grad():
            for p, d in zip(self.net.weights(), dw):
                p += eps * d

        hessian = [(p - n) / 2.0 * eps for p, n in zip(dalpha_pos, dalpha_neg)]
        return hessian

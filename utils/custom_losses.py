"""
Class defnition for training loss functions
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings

class ALMTermsClass(nn.Module):
    """
    Class to calculate Augmented Lagrangian Method (ALM) terms for loss function.
    """
    def __init__(self,delta, p2_norm, device):
        super(ALMTermsClass, self).__init__()
        """
            delta (float): Delta parameter for constraint.
            mu (float): Penalty parameter for ALM.
            p2_norm (int): Exponent for the L2 norm in the constraint."""
        
        self.delta = delta
        self.p2_norm = p2_norm
        self.device = device
    def forward(
        self,
        buffer_batch_pos,
        buffer_batch_neg,
        lambdas_index_buffer,
        lambdas,
        mu
        ):
        """
        Calculate Augmented Lagrangian Method (ALM) terms for loss function.

        Parameters:
            buffer_batch_pos (list): List of positive samples.
            buffer_batch_neg (list): List of negative samples.
            lambdas_index_buffer (list): List of lambda index values.
            lambdas (list): List of lambda values.
            device (torch.device): Device to perform computations (e.g., 'cuda' or 'cpu').

        Returns:
            tuple: A tuple containing:
                - Updated lambdas (list).
                - Loss (torch.Tensor)."""
        relu_function = nn.ReLU()
        for pos_sample, single_lambda_index in zip(buffer_batch_pos, lambdas_index_buffer):
            q_pos = torch.zeros([1, 1]).to(self.device)
            for neg_sample in buffer_batch_neg:
                q_pos += relu_function(-(pos_sample - neg_sample) + self.delta)
            loss = (
                mu / 2 * torch.pow(q_pos, 2 * self.p2_norm)
                + lambdas[int(single_lambda_index)] * q_pos
            ) / (len(buffer_batch_pos) * len(buffer_batch_neg))
            # Update Lambda
            lambdas[int(single_lambda_index)] += mu * q_pos
        return lambdas, loss


class SymBCEFocalLoss(nn.Module):
    """
    Symmetric BCE focal loss
    """

    def __init__(self, gamma):
        super(SymBCEFocalLoss, self).__init__()
        self.gamma = gamma

    def asym_focal_loss(self, x, t):  # x = NN output, t = 0/1 label
        FL = torch.zeros_like(t, dtype=float)
        FL[(t).nonzero()] = (
            -torch.pow(1 - torch.sigmoid(x[(t).nonzero()]), self.gamma)
            * torch.log(torch.sigmoid(x[(t).nonzero()]) + 1e-4).double()
        )
        FL[(t == 0).nonzero()] = (
            -torch.pow(torch.sigmoid(x[(t == 0).nonzero()]), self.gamma)
            * torch.log(1 - torch.sigmoid(x[(t == 0).nonzero()]) + 1e-4).double()
        )
        return torch.mean(FL)

    def forward(self, logits, targets):
        FL = self.asym_focal_loss(logits, targets)
        return FL


class SymBCELargeMargin(nn.Module):
    """
    Symmetric BCE margin loss loss
    """

    def __init__(self, m):
        super(SymBCELargeMargin, self).__init__()
        self.m = m

    def sym_large_margin_bce(self, x, t):  # x = NN output, t = 0/1 label
        LM = torch.zeros_like(t, dtype=float)
        LM[(t).nonzero()] = -torch.log(
            torch.sigmoid(x[(t).nonzero()] - self.m) + 1e-4
        ).double()
        LM[(t == 0).nonzero()] = -torch.log(
            1 - torch.sigmoid(x[(t == 0).nonzero()] + self.m) + 1e-4
        ).double()
        # print("check", torch.sigmoid(x-m), 1-torch.sigmoid(x), t)
        return torch.mean(LM)

    def forward(self, logits, targets):
        LM = self.sym_large_margin_bce(logits, targets)
        return LM


class AsymBCEFocalLoss(nn.Module):
    """
    Asymmetric BCE focal loss
    """

    def __init__(self, gamma):
        super(AsymBCEFocalLoss, self).__init__()
        self.gamma = gamma

    def asym_focal_loss(self, x, t):  # x = NN output, t = 0/1 label
        FL = torch.zeros_like(t, dtype=float)
        FL[(t).nonzero()] = -torch.log(torch.sigmoid(x[(t).nonzero()]) + 1e-4).double()
        FL[(t == 0).nonzero()] = (
            -torch.pow(torch.sigmoid(x[(t == 0).nonzero()]), self.gamma)
            * torch.log(1 - torch.sigmoid(x[(t == 0).nonzero()]) + 1e-4).double()
        )
        return torch.mean(FL)

    def forward(self, logits, targets):
        FL = self.asym_focal_loss(logits, targets)
        return FL


class AsymBCELargeMargin(nn.Module):
    """
    Asymmetric BCE margin loss loss
    """

    def __init__(self, m):
        super(AsymBCELargeMargin, self).__init__()
        self.m = m

    def asym_large_margin_bce(self, x, t):  # x = NN output, t = 0/1 label
        LM = torch.zeros_like(t, dtype=float)
        LM[(t).nonzero()] = -torch.log(
            torch.sigmoid(x[(t).nonzero()] - self.m) + 1e-4
        ).double()
        LM[(t == 0).nonzero()] = -torch.log(
            1 - torch.sigmoid(x[(t == 0).nonzero()]) + 1e-4
        ).double()
        return torch.mean(LM)

    def forward(self, logits, targets):
        LM = self.asym_large_margin_bce(logits, targets)
        return LM


class WeightedBCE(nn.Module):
    """
    Weighted BCE
    """

    def __init__(self, c):
        super(WeightedBCE, self).__init__()
        self.c = c

    def weighted_loss(self, x, t):  # x = NN output, t = 0/1 label
        BCE = torch.nn.BCEWithLogitsLoss(reduction="none")
        weight_tensor = torch.ones_like(t, dtype=float)
        weight_tensor[(t).nonzero()] = self.c
        # print(x, weight_tensor, x*weight_tensor)
        weighted_BCE = torch.mean(weight_tensor * BCE(x, t))
        return weighted_BCE

    def forward(self, logits, targets):
        weighted_BCE = self.weighted_loss(logits, targets)
        return weighted_BCE


class ClassBalancedBCE(nn.Module):
    """
    Class balanced BCE
    """

    def __init__(self, beta, n_0, n_1):
        super(ClassBalancedBCE, self).__init__()
        self.beta = beta
        self.n_0 = n_0
        self.n_1 = n_1

    def class_balanced_loss(self, x, t):  # x = NN output, t = 0/1 label
        BCE = torch.nn.BCEWithLogitsLoss(reduction="none")
        weight_tensor = torch.zeros_like(t, dtype=float)
        weight_tensor[(t).nonzero()] = (1 - self.beta) / (1 - self.beta**self.n_1)
        weight_tensor[(t == 0).nonzero()] = (1 - self.beta) / (
            1 - self.beta**self.n_0
        )
        # print(x, weight_tensor, x*weight_tensor)
        cb_BCE = torch.mean(weight_tensor * BCE(x, t))
        return cb_BCE

    def forward(self, logits, targets):
        cb_BCE = self.class_balanced_loss(logits, targets)
        return cb_BCE


class MiniBatchAUC(nn.Module):
    """
    Mini batch AUC
    """

    def __init__(self):
        super(MiniBatchAUC, self).__init__()

    def mini_batch_auc(self, x, t):  # x = NN output, t = 0/1 label
        # x_pos = x[(t).nonzero()]
        # x_neg = x[(t==0).nonzero()]
        if x[(t).nonzero()].size() != 0:
            for i, pos_sample in enumerate(x[(t).nonzero()]):
                # for j, neg_sample in enumerate(x[(t==0).nonzero()]):
                if i == 0:
                    loss = sum(
                        torch.pow(
                            1 - (torch.sigmoid(pos_sample) - torch.sigmoid(x[t == 0])),
                            2,
                        )
                    )
                else:
                    loss += sum(
                        torch.pow(
                            1 - (torch.sigmoid(pos_sample) - torch.sigmoid(x[t == 0])),
                            2,
                        )
                    )
            if len(x[(t).nonzero()]) != 0 and len(x[(t == 0).nonzero()]) != 0:
                loss = loss / (len(x[(t).nonzero()]) * len(x[(t == 0).nonzero()]))
                return torch.sum(loss)

    def forward(self, logits, targets):
        MBAUC = self.mini_batch_auc(logits, targets)
        return MBAUC


class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        # store as buffer so it moves with the module/device
        self.register_buffer("m_list", torch.tensor(m_list, dtype=torch.float))
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.bool)
        index.scatter_(1, target.data.view(-1, 1).long(), True)

        index_float = index.float().to(x.device)
        batch_m = torch.matmul(self.m_list[None, :].to(x.device), index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m

        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s * output, target.long(), weight=self.weight)


def select_objective_function(
    training_type, gamma, m, c, ratio_pos_train, num_maj, beta, delta, cls_num_list
):
    """
    Function to selct the loss function to be used as objective.
    """
    if training_type == "ASYM_FL":
        loss_function = AsymBCEFocalLoss(gamma)
    elif training_type == "ASYM_LM":
        loss_function = AsymBCELargeMargin(m)
    elif training_type == "SYM_FL":
        loss_function = SymBCEFocalLoss(gamma)
    elif training_type == "SYM_LM":
        loss_function = SymBCELargeMargin(m)
    elif training_type == "WBCE":
        loss_function = WeightedBCE(c * ratio_pos_train)
    elif training_type == "cb_BCE":
        n_0 = num_maj
        n_1 = int(num_maj / ratio_pos_train)
        loss_function = ClassBalancedBCE(beta, n_0, n_1)
    elif training_type == "MBAUC":
        loss_function = MiniBatchAUC()
    elif training_type == "LDAM":
        loss_function = LDAMLoss(
            cls_num_list=cls_num_list, max_m=0.5, s=30, weight=None
        )
    elif training_type == "BCE":
        loss_function = torch.nn.BCEWithLogitsLoss()
    else:
        warnings.warn("Loss type is not listed")

    return loss_function

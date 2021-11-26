import torch

from .losses import studentT, error_type_determination
from .losses import _get_matcher



def occ_csi_soft_loss(distances, target_labels, prototype_labels, theta_boundary, device='cpu'):
    """
    OneClassClassifier loss function implemented with Student-t distribution
    """
    # get probabilty from distribution
    prob = studentT(distances, theta_boundary)

    # filter FP, FN
    TP, TN, FP, FN = error_type_determination(distances, theta_boundary, target_labels, prototype_labels, device) 

    # calc loss
    TPloss = (TP * prob)
    FPloss = (FP * prob)
    FNloss = 1 - (FN * prob)

    csi = (TPloss + 1) / (FNloss + FPloss + TPloss)
    csi_orig = (TP.sum()) / (FN.sum() + FP.sum() + TP.sum())
    print("csi score:",csi_orig)
    #print(csi.mean())
    loss = 1 / csi

    return loss.mean()


def occ_brier_score(distances, target_labels, prototype_labels, theta_boundary, device='cpu'):
    """
    OneClassClassifier loss function implemented with Student-t distribution
    """
    # get probabilty from distribution
    prob = studentT(distances, theta_boundary)

    # calc loss
    c = (torch.amax(prototype_labels) + 1) - target_labels
    zero = torch.Tensor([[0]])
    print(zero)
    print(studentT(zero, theta_boundary))
    loss = (prob - c.float()) ** 2
    print("brier score:",loss.mean())

    return loss.mean()


def occ_heidke_skill_score(distances, target_labels, prototype_labels, theta_boundary, device='cpu'):
    """
    OneClassClassifier loss function implemented with Student-t distribution
    """
    # get probabilty from distribution
    prob = studentT(distances, theta_boundary)

    # filter FP, FN
    TP, TN, FP, FN = error_type_determination(distances, theta_boundary, target_labels, prototype_labels, device) 
    
    # calc loss
    TPloss = (TP * prob).sum()
    FPloss = (FP * prob).sum()
    TNloss = (1 - (TN * prob)).sum()
    FNloss = (1 - (FN * prob)).sum()

    nominator = 2 * (TPloss * TNloss - FPloss * FNloss)
    denominator = (TPloss + FNloss) * (FNloss + TNloss) + (TPloss + FPloss) * (FPloss + TNloss)
    
    T = nominator / denominator
    loss = 1/T

    return loss.mean()

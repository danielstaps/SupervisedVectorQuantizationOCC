import torch

from .losses import error_type_determination
from .losses import _get_matcher
from .distributions import studentT


def occ_csi_soft_loss(distances, target_labels, prototype_labels, theta_boundary, device='cpu'):
    """
    OneClassClassifier loss function implemented with Student-t distribution
    """
    # get probabilty from distribution
    prob = studentT(distances, theta_boundary)
    #print("probs",prob)
    
    # filter FP, FN
    TP, TN, FP, FN = error_type_determination(distances, theta_boundary, target_labels, prototype_labels, device)

    # calc loss
    TPloss = (TP.to(device) * prob.to(device))
    FPloss = (FP.to(device) * prob.to(device))
    FNloss = 1 - (FN.to(device) * prob.to(device))
    #print("conf",TPloss, FPloss, FNloss)

    csi = (TPloss + 1) / (FNloss + FPloss + TPloss)
    #print("csi",csi)
    #csi_orig = (TP.sum()) / (FN.sum() + FP.sum() + TP.sum())
    #print("csi score:",csi_orig)
    #print(csi.mean())
    loss = 1 / csi

    return loss.mean()


def occ_brier_score(distances, target_labels, prototype_labels, theta_boundary, device='cpu'):
    """
    OneClassClassifier loss function implemented with Student-t distribution
    """
    # get probabilty from distribution
    prob = studentT(distances, theta_boundary,
            idx=target_labels, prototype_labels=prototype_labels)

    # calc loss
    #c = (torch.amax(prototype_labels) + 1) - target_labels
    #c = torch.where(c == 0, 0, 1)
    c = torch.where(target_labels > torch.amax(prototype_labels), 0, 1)

    #print(prob)
    #print(prob/norm_scalar)
    loss = (prob - c.float()) ** 2
    #print("brier score:",loss.mean())

    return loss.mean()

def occ_brier_score(distances, target_labels, prototype_labels, theta_boundary, device='cpu'):
    """
    OneClassClassifier loss function implemented with Student-t distribution
    """
    # get probabilty from distribution
    prob = studentT(distances, theta_boundary,
            idx=target_labels, prototype_labels=prototype_labels)

    # calc loss
    c = torch.where(target_labels > torch.amax(prototype_labels), 0, 1)


    loss = (prob - c.float()) ** 2
    #print("brier score:",loss.mean())

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

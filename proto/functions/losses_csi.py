import torch

from .losses import error_type_determination
from .losses import _get_matcher
from .distributions import distribution_handler

from sklearn.metrics import confusion_matrix


def occ_csi_soft_loss(distances, target_labels, prototype_labels, theta_boundary, device='cpu'):
    """
    OneClassClassifier loss function implemented with Student-t distribution
    """
    # get probabilty from distribution
    prob = distribution_handler(distances, theta_boundary)
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
    prob = distribution_handler(distances, theta_boundary, distribution='studentT',
            idx=target_labels, prototype_labels=prototype_labels)

    # calc loss
    #c = (torch.amax(prototype_labels) + 1) - target_labels
    #c = torch.where(c == 0, 0, 1)
    c = torch.where(target_labels > torch.amax(prototype_labels), 0, 1)

    #print(prob)
    #print(prob/norm_scalar)
    loss = (prob - c.float()) ** 2
    print("brier score:",loss.mean())

    """
    d_tilde = torch.subtract(distances.to(device), theta_boundary.to(device))
    is_out_of_bound = d_tilde >= 0
    print(confusion_matrix(target_labels.cpu().numpy(), is_out_of_bound.cpu().numpy()))
    """
    return loss.mean()

def occ_brier_score2(distances, target_labels, prototype_labels, theta_boundary, device='cpu'):
    """
    OneClassClassifier loss function implemented with Student-t distribution
    """
    matcher = _get_matcher(target_labels, prototype_labels, device=device)
    #not_matcher = torch.bitwise_not(matcher)

    # get probabilty from distribution
    prob = distribution_handler(distances, theta_boundary, pass_probs=True, distribution='gauss')
    # props from protos with minimum distance to data


    classes = torch.unique(prototype_labels)
    num_classes = classes.shape[0]
    p = torch.zeros(size=(distances.shape[0], num_classes))
    c = torch.zeros(size=(distances.shape[0], num_classes))
    for i in classes:
        protoii = torch.eq(i, prototype_labels)
        selected_distances = distances[:,protoii]
        selected_probs = prob[:,protoii]
        winning_indices = torch.min(selected_distances, dim=1).indices.to(device) # list of winning prototypes
        p[:, i] = selected_probs.gather(1, winning_indices.unsqueeze(1)).squeeze()
        c[:, i] = torch.where(target_labels == i, 1, 0)

    #                       prototypes = [      0       ,        1        ]
    # target_labels = [0, 1, 0, 1, 1] -> [[1, 0, 1, 0, 0], [0, 1, 0, 1, 1]]
    # target_labels = [2, 1, 0, 2, 1] -> [[0, 0, 1, 0, 0], [0, 1, 0, 0, 1], [1, 0, 0, 1, 0]]

    #                       prototypes = [      0       ,        0        ,       1        ]
    # target_labels = [0, 1, 0, 1, 1] -> [[1, 0, 1, 0, 0], [1, 0, 1, 0, 0], [0, 1, 0, 1, 1]]
    # => falls kollabiert -> winning proto per class, oder Superposition

    loss = (p - c) ** 2
    print("brier score:",loss.mean())

    return loss.mean()


def occ_heidke_skill_score(distances, target_labels, prototype_labels, theta_boundary, device='cpu'):
    """
    OneClassClassifier loss function implemented with Student-t distribution
    """
    # get probabilty from distribution
    prob = distribution_handler(distances, theta_boundary)

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

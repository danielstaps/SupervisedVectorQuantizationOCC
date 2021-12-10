import torch

from .confusion import error_type_determination
from .distributions import distribution_handler


def occ_csi_soft_loss(
    distances,
    target_labels,
    prototype_labels,
    theta_boundary,
):
    """
    OneClassClassifier loss function implemented with Student-t distribution
    """
    prob = distribution_handler(
        distances,
        theta_boundary,
        distribution='studentT',
    )

    # filter FP, FN
    TP, TN, FP, FN = error_type_determination(
        distances,
        theta_boundary,
        target_labels,
        prototype_labels,
    )

    # calc loss
    TPloss = (TP * prob)
    FPloss = (FP * prob)
    FNloss = 1 - (FN * prob)
    #print("conf",TPloss, FPloss, FNloss)

    TPloss = torch.where(TPloss <= torch.Tensor([[1e-4]]),
                         torch.tensor([[1e-4]]), TPloss)

    csi = (TPloss) / (FNloss + FPloss + TPloss)
    #print("csi",csi)
    csi_orig = (TP.sum()) / (FN.sum() + FP.sum() + TP.sum())
    #print("csi score:", csi_orig)
    #print("csi mean:", csi.mean().detach())
    loss = 1 / csi

    return loss.mean()


def occ_csi_soft_loss2(
    distances,
    target_labels,
    prototype_labels,
    theta_boundary,
):
    """
    OneClassClassifier loss function implemented with Student-t distribution
    """
    # get probabilty from distribution
    prob = distribution_handler(
        distances,
        theta_boundary,
        distribution='studentT',
        pass_probs=True,
    )

    # filter FP, FN
    TP, TN, FP, FN = error_type_determination(
        distances,
        theta_boundary,
        target_labels,
        prototype_labels,
        pass_errors=True,
    )

    # calc loss
    TPloss = (TP * prob)
    FPloss = (FP * prob)
    FNloss = 1 - (FN * prob)
    #print("conf", TPloss.detach(), FPloss.detach(), FNloss.detach())

    TPloss = torch.clip(TPloss, min=1e-4)

    csi = (TPloss) / (FNloss + FPloss + TPloss)

    classes = torch.unique(prototype_labels)
    num_classes = classes.shape[0]
    local_loss = torch.zeros(size=(distances.shape[0],
                                   num_classes)).type_as(distances)
    for i in classes:
        protoii = torch.eq(i, prototype_labels)
        selected_distances = distances[:, protoii]
        winning_indices = torch.min(selected_distances, dim=1).indices
        local_loss[:, i] = csi[:, protoii].gather(
            1, winning_indices.unsqueeze(1)).squeeze()

    csi = local_loss
    #print("csi",csi)
    csi_orig = (TP.sum()) / (FN.sum() + FP.sum() + TP.sum())
    #print("csi score:", csi_orig)
    #print("csi mean:", csi.mean().detach())
    loss = 1 / csi

    return loss.mean()


def occ_brier_score(
    distances,
    target_labels,
    prototype_labels,
    theta_boundary,
):
    """
    OneClassClassifier loss function implemented with Student-t distribution
    """
    print(theta_boundary)
    # get probabilty from distribution
    prob = distribution_handler(distances,
                                theta_boundary,
                                distribution='studentT',
                                idx=target_labels,
                                prototype_labels=prototype_labels)

    # calc loss
    #c = (torch.amax(prototype_labels) + 1) - target_labels
    #c = torch.where(c == 0, 0, 1)
    c = torch.where(target_labels > torch.amax(prototype_labels), 0, 1)

    #print(prob)
    #print(prob/norm_scalar)
    loss = (prob - c.float())**2
    print("brier score:", loss.mean())
    """
    d_tilde = torch.subtract(distances, theta_boundary)
    is_out_of_bound = d_tilde >= 0
    print(confusion_matrix(target_labels.cpu().numpy(), is_out_of_bound.cpu().numpy()))
    """
    return loss.mean()


def occ_brier_score2(
    distances,
    target_labels,
    prototype_labels,
    theta_boundary,
):
    """
    OneClassClassifier loss function implemented with Student-t distribution
    """
    # get probabilty from distribution
    prob = distribution_handler(
        distances,
        theta_boundary,
        pass_probs=True,
        distribution='studentT',
    )
    # props from protos with minimum distance to data

    classes = torch.unique(prototype_labels)
    num_classes = classes.shape[0]

    local_loss = torch.zeros(size=(num_classes, ))
    for i in classes:
        protoii = torch.eq(i, prototype_labels)
        selected_distances = distances[:, protoii]
        selected_probs = prob[:, protoii]
        winning_indices = torch.min(
            selected_distances, dim=1).indices  # list of winning prototypes
        p = selected_probs.gather(1, winning_indices.unsqueeze(1)).squeeze()
        c = torch.where(target_labels == i, 1, 0)

        local_loss[i] = ((p - c)**2).mean()

    loss = local_loss

    return loss.mean()

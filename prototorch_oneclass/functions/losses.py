import torch
from prototorch.core.losses import NeuralGasEnergy

from .confusion import error_type_determination
from .distributions import get_probabilities, sigmoid


def csi_soft_loss(
    distances,
    target_labels,
    prototype_labels,
    theta_boundary,
    distribution=None,
    sigma=0.1,
):
    """
    OneClassClassifier loss function implemented with Student-t distribution
    """

    if distribution is None:
        distribution = 'studentT'

    prob = get_probabilities(
        distances,
        theta_boundary,
        distribution=distribution,
    )

    tp, _, fp, fn = error_type_determination(
        distances,
        target_labels,
        prototype_labels,
        theta_boundary,
    )

    trick17 = prob * sigmoid(theta_boundary - distances, sigma)

    tpLoss = tp * trick17
    fpLoss = fp * trick17
    fnLoss = 1 - (fn * trick17)

    tpLoss = torch.clip(tpLoss, min=1e-4)

    #csi = (tpLoss) / (fnLoss + fpLoss + tpLoss)
    csi = (tpLoss.mean(dim=1)) / (fnLoss.mean(dim=1) + fpLoss.mean(dim=1) +
                                  tpLoss.mean(dim=1))
    print(csi.shape)
    """
    classes = torch.unique(prototype_labels)
    num_classes = classes.shape[0]
    local_loss = torch.zeros(size=(distances.shape[0],
                                   num_classes)).type_as(distances)
    for i in classes:
        protoii = torch.eq(i, prototype_labels)
        selected_distances = distances[:, protoii]
        winning_indices = torch.min(selected_distances, dim=1).indices
        local_loss[:, i] = csi[:, protoii].gather(
            1,
            winning_indices.unsqueeze(1),
        ).squeeze()

    csi = local_loss
    """
    loss = 1 / csi

    return loss.mean()


def lpcsi_loss(
    distances,
    target_labels,
    prototype_labels,
    theta_boundary,
    distribution=None,
    sigma=0.1,
):
    """
    OneClassClassifier loss function implemented with Student-t distribution
    """

    if distribution is None:
        distribution = 'studentT'

    prob = get_probabilities(
        distances,
        theta_boundary,
        distribution=distribution,
    )

    heavy_in_side = sigmoid(theta_boundary - distances, sigma)
    heavy_out_side = sigmoid(distances - theta_boundary, sigma)

    kronecker_delta_plus = torch.where(target_labels == 0, 1, 0)
    kronecker_delta_minus = torch.logical_not(kronecker_delta_plus)

    tpLoss = kronecker_delta_plus * prob * heavy_in_side
    fpLoss = kronecker_delta_minus * prob * heavy_in_side
    fnLoss = kronecker_delta_plus * (1 - prob) * heavy_out_side
    #fnLoss = kronecker_delta_plus * (1 - prob) * heavy_in_side

    tpLoss = torch.sum(tpLoss, dim=1)
    fpLoss = torch.sum(fpLoss, dim=1)
    fnLoss = torch.sum(fnLoss, dim=1)
    #tpLoss = torch.clip(tpLoss, min=1e-4)

    csi = (tpLoss) / (fnLoss + fpLoss + tpLoss)
    """
    classes = torch.unique(prototype_labels)
    num_classes = classes.shape[0]
    local_loss = torch.zeros(size=(distances.shape[0],
                                   num_classes)).type_as(distances)
    for i in classes:
        protoii = torch.eq(i, prototype_labels)
        selected_distances = distances[:, protoii]
        winning_indices = torch.min(selected_distances, dim=1).indices
        local_loss[:, i] = csi[:, protoii].gather(
            1,
            winning_indices.unsqueeze(1),
        ).squeeze()
    """

    #csi = local_loss
    classification_loss = 1 / csi

    representation_loss, _ = NeuralGasEnergy(lm=1)(distances)

    #return loss.mean()
    alpha = 0.5

    return alpha * classification_loss.mean() + (
        1 - alpha) * representation_loss.mean()


def brier_score(
    distances,
    target_labels,
    prototype_labels,
    theta_boundary,
    distribution=None,
):
    """
    OneClassClassifier loss function implemented with Student-t distribution
    """

    if distribution is None:
        distribution = 'studentT'

    prob = get_probabilities(
        distances,
        theta_boundary,
        distribution=distribution,
    )

    classes = torch.unique(prototype_labels)
    num_classes = classes.shape[0]

    local_loss = torch.zeros(size=(num_classes, ))
    for i in classes:
        protoii = torch.eq(i, prototype_labels)
        selected_distances = distances[:, protoii]
        selected_probs = prob[:, protoii]
        winning_indices = torch.min(
            selected_distances,
            dim=1,
        ).indices
        p = selected_probs.gather(1, winning_indices.unsqueeze(1)).squeeze()
        c = torch.where(target_labels == i, 1, 0)

        local_loss[i] = ((p - c)**2).mean()

    loss = local_loss

    return loss.mean()

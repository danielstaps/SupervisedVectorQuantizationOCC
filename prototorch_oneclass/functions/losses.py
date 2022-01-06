import torch

from .confusion import boundary_distance, error_type_determination
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

    d_tilde = boundary_distance(distances, theta_boundary)

    trick17 = prob * sigmoid(d_tilde, sigma)

    tpLoss = tp * trick17
    fpLoss = fp * trick17
    fnLoss = 1 - (fn * trick17)

    tpLoss = torch.clip(tpLoss, min=1e-4)

    csi = (tpLoss) / (fnLoss + fpLoss + tpLoss)

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
    loss = 1 / csi

    return loss.mean()


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

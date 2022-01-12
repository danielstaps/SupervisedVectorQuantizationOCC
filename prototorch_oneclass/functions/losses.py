import torch
from prototorch.core.losses import NeuralGasEnergy

from .confusion import error_type_determination, get_scores
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
    score=None,
    gamma=0.1,
    sigma=0.1,
):
    """
    OneClassClassifier loss function implemented with Student-t distribution
    """

    if distribution is None:
        distribution = 'studentT'

    if score is None:
        score = 'csi_score'

    #prob = get_probabilities(
    #    distances,
    #    gamma,
    #    distribution=distribution,
    #)

    prob = torch.where(distances < theta_boundary, 1, 0)

    heavyside = sigmoid(theta_boundary - distances, sigma)

    kronecker_delta_plus = torch.where(target_labels == 0, 1, 0)
    if len(kronecker_delta_plus.shape) < 2:
        kronecker_delta_plus = torch.unsqueeze(kronecker_delta_plus, 1)

    tpLoss = kronecker_delta_plus * heavyside * prob
    tnLoss = (1 - kronecker_delta_plus) * (1 - heavyside * prob)
    fpLoss = (1 - kronecker_delta_plus) * heavyside * prob
    fnLoss = kronecker_delta_plus * (1 - heavyside * prob)

    tpLoss = torch.clip(tpLoss, min=1e-4)

    classes = torch.unique(prototype_labels)
    num_classes = classes.shape[0]

    tp_local = torch.zeros(size=(distances.shape[0],
                                 num_classes)).type_as(distances)
    tn_local = torch.zeros(size=(distances.shape[0],
                                 num_classes)).type_as(distances)
    fp_local = torch.zeros(size=(distances.shape[0],
                                 num_classes)).type_as(distances)
    fn_local = torch.zeros(size=(distances.shape[0],
                                 num_classes)).type_as(distances)

    for i in classes:
        protoii = torch.eq(i, prototype_labels)
        selected_distances = distances[:, protoii]
        winning_indices = torch.min(selected_distances, dim=1).indices
        tp_local[:, i] = tpLoss[:, protoii].gather(
            1,
            winning_indices.unsqueeze(1),
        ).squeeze()
        tn_local[:, i] = tnLoss[:, protoii].gather(
            1,
            winning_indices.unsqueeze(1),
        ).squeeze()
        fp_local[:, i] = fpLoss[:, protoii].gather(
            1,
            winning_indices.unsqueeze(1),
        ).squeeze()
        fn_local[:, i] = fnLoss[:, protoii].gather(
            1,
            winning_indices.unsqueeze(1),
        ).squeeze()

    scores = get_scores(score, tp_local, tn_local, fp_local, fn_local)

    classification_loss = scores
    representation_loss, _ = NeuralGasEnergy(lm=1)(distances)

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

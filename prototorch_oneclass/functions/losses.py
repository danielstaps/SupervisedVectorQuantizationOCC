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
    score=None,
    gamma=0.1,
    sigma=0.1,
    backbone=None,
):
    """
    OneClassClassifier loss function implemented with Student-t distribution
    """

    if backbone is not None:
        print(theta_boundary)
        print(backbone.detach().cpu())

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

    trick17 = prob  #* sigmoid(theta_boundary - distances, sigma)

    tpLoss = tp * trick17
    fpLoss = fp * trick17
    fnLoss = 1 - (fn * trick17)

    tpLoss = torch.clip(tpLoss, min=1e-4)

    csi = (tpLoss) / (fnLoss + fpLoss + tpLoss)
    #csi = (tpLoss.mean(dim=1)) / (fnLoss.mean(dim=1) + fpLoss.mean(dim=1) +
    #                              tpLoss.mean(dim=1))

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


def lpcsi_loss(distances,
               target_labels,
               prototype_labels,
               theta_boundary,
               distribution=None,
               score=None,
               gamma=0.1,
               sigma=0.1,
               backbone=None):
    """
    OneClassClassifier loss function implemented with Student-t distribution
    """

    if distribution is None:
        distribution = 'gauss'

    if score is None:
        score = 'pcs'

    prob = get_probabilities(
        distances,
        #gamma,
        theta_boundary,
        distribution=distribution,
    )
    #prob = torch.where(distances < theta_boundary, 1, 0)

    heavyside = sigmoid(theta_boundary - distances, sigma)

    kronecker_delta_plus = torch.where(target_labels == 0, 1, 0)
    if len(kronecker_delta_plus.shape) < 2:
        kronecker_delta_plus = torch.unsqueeze(kronecker_delta_plus, 1)

    r = heavyside * prob

    tpLoss = kronecker_delta_plus * r
    tnLoss = (1 - kronecker_delta_plus) * (1 - r)
    fpLoss = (1 - kronecker_delta_plus) * r
    fnLoss = kronecker_delta_plus * (1 - r)

    if score == 'csi':
        tpLoss = torch.clip(tpLoss, min=1e-4)

    classes = torch.unique(prototype_labels)
    num_classes = classes.shape[0]

    local_scores = get_scores(score, tpLoss, tnLoss, fpLoss, fnLoss)
    scores = torch.zeros(size=(distances.shape[0],
                               num_classes)).type_as(distances)
    for i in classes:
        protoii = torch.eq(i, prototype_labels)
        selected_distances = distances[:, protoii]
        winning_indices = torch.min(selected_distances, dim=1).indices
        scores[:, i] = local_scores[:, protoii].gather(
            1,
            winning_indices.unsqueeze(1),
        ).squeeze()

    #classification_loss = 1 / scores
    classification_loss = -scores
    representation_loss, _ = NeuralGasEnergy(lm=1)(
        distances[target_labels == 0, :])

    alpha = 0.2
    return alpha * representation_loss.mean() + (
        1 - alpha) * classification_loss.mean()


def occ_entropy_loss(distances,
                     target_labels,
                     prototype_labels,
                     theta_boundary,
                     distribution=None,
                     score=None,
                     gamma=0.1,
                     sigma=0.1,
                     backbone=None):
    """
    OneClassClassifier loss function implemented with Student-t distribution
    """

    if distribution is None:
        distribution = 'studentT'

    prob = get_probabilities(
        distances,
        #gamma,
        theta_boundary,
        distribution=distribution,
    )
    #prob = torch.where(distances < theta_boundary, 1, 0)

    heavyside = sigmoid(theta_boundary - distances, sigma)
    target_class_plus = torch.where(target_labels == 0, 1, 0)
    if len(target_class_plus.shape) < 2:
        target_class_plus = torch.unsqueeze(target_class_plus, 1)

    r = heavyside * prob
   
    epsilon = 1e-10

    tpLoss = target_class_plus * torch.log(r + epsilon)
    tnLoss = (1 - target_class_plus) * torch.log((1 - r) + epsilon)

    local_ce = -(tpLoss + tnLoss)

    classes = torch.unique(prototype_labels)
    num_classes = classes.shape[0]

    win_ce = torch.zeros(size=(distances.shape[0],
                               num_classes)).type_as(distances)
    class_ng = torch.zeros(size=(num_classes, )).type_as(distances)

    for i in classes:
        # classification
        protoii = torch.eq(i, prototype_labels)
        selected_distances = distances[:, protoii]
        winning_indices = torch.min(selected_distances, dim=1).indices
        win_ce[:, i] = local_ce[:, protoii].gather(
            1,
            winning_indices.unsqueeze(1),
        ).squeeze()
        # representation
        class_ng_loss, _ = NeuralGasEnergy(lm=1)(
            distances[target_labels == i, :])
        class_ng[i] = class_ng_loss

    classification_loss = win_ce
    representation_loss = class_ng

    alpha = 0.2
    return alpha * representation_loss.mean() + (
        1 - alpha) * classification_loss.mean()


def brier_score(
    distances,
    target_labels,
    prototype_labels,
    theta_boundary,
    distribution=None,
    score=None,
    gamma=0.1,
    sigma=0.1,
    backbone=None
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
    heavyside = sigmoid(theta_boundary - distances, sigma)
    r = heavyside * prob
   
    classes = torch.unique(prototype_labels)
    num_classes = classes.shape[0]

    local_loss = torch.zeros(size=(num_classes, ))
    class_ng = torch.zeros(size=(num_classes, )).type_as(distances)
    for i in classes:
        protoii = torch.eq(i, prototype_labels)
        selected_distances = distances[:, protoii]
        selected_r = r[:, protoii]
        winning_indices = torch.min(
            selected_distances,
            dim=1,
        ).indices
        r_win = selected_r.gather(1, winning_indices.unsqueeze(1)).squeeze()
        c = torch.where(target_labels == i, 1, 0)

        local_loss[i] = ((r_win - c)**2).mean()
        # representation
        class_ng_loss, _ = NeuralGasEnergy(lm=1)(
            distances[target_labels == i, :])
        class_ng[i] = class_ng_loss

    classification_loss = local_loss
    representation_loss = class_ng

    alpha = 0.2
    return alpha * representation_loss.mean() + (
        1 - alpha) * classification_loss.mean()

import torch


def _get_matcher(targets, labels):
    """Returns a boolean tensor."""
    matcher = torch.eq(targets.unsqueeze(dim=1), labels)
    if labels.ndim == 2:
        # if the labels are one-hot vectors
        num_classes = targets.size()[1]
        matcher = torch.eq(torch.sum(matcher, dim=-1), num_classes)
    return matcher


def error_type_determination(
    distances,
    target_labels,
    prototype_labels,
    theta_boundary,
):
    matcher = _get_matcher(target_labels, prototype_labels)
    not_matcher = torch.bitwise_not(matcher)

    d_tilde = distances - theta_boundary

    is_in_bound = d_tilde < 0
    is_out_of_bound = d_tilde >= 0

    tp = torch.logical_and(is_in_bound, matcher)
    fn = torch.logical_and(is_out_of_bound, matcher)
    tn = torch.logical_and(is_out_of_bound, not_matcher)
    fp = torch.logical_and(is_in_bound, not_matcher)

    return tp, tn, fp, fn


def csi_score(tpLoss, tnLoss, fpLoss, fnLoss):
    csi = (tpLoss) / (fnLoss + fpLoss + tpLoss)
    return csi


def mod_csi_score(tpLoss, tnLoss, fpLoss, fnLoss):
    csi = (tpLoss + tnLoss) / (fnLoss + fpLoss + tpLoss)
    return csi


def accuracy_score(tpLoss, tnLoss, fpLoss, fnLoss):
    accuracy = (tpLoss + tnLoss) / (tpLoss + tnLoss + fpLoss + fnLoss)
    return accuracy


SCORES = {
    "csi": csi_score,
    "mod_csi": mod_csi_score,
    "accuracy": accuracy_score,
}


def get_scores(score, tpLoss, tnLoss, fpLoss, fnLoss):

    if score not in SCORES:
        raise ValueError(
            f"Unknown distribution {score} for distribution_handler, choose from {list(SCORES.keys())}"
        )

    return SCORES[score](tpLoss, tnLoss, fpLoss, fnLoss)

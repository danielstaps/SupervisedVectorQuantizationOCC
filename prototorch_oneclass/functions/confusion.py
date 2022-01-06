import torch


def boundary_distance(distances, theta_boundary):
    return distances - theta_boundary


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

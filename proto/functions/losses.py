"""ProtoTorch loss functions."""

import torch


def _get_matcher(targets, labels, device='cpu'):
    """Returns a boolean tensor."""
    #matcher = torch.eq(targets.unsqueeze(dim=1), labels)
    targets_resized = targets.unsqueeze(dim=1)
    matcher = torch.eq(targets_resized, labels.to(device))
    if labels.ndim == 2:
        # if the labels are one-hot vectors
        num_classes = targets.size()[1]
        matcher = torch.eq(torch.sum(matcher, dim=-1), num_classes)
    return matcher


def _get_dp_dm(distances, targets, plabels, with_indices=False, fill_value=float("inf")):
    """Returns the d+ and d- values for a batch of distances."""
    matcher = _get_matcher(targets, plabels)
    not_matcher = torch.bitwise_not(matcher)

    inf = torch.full_like(distances, fill_value=fill_value)
    d_matching = torch.where(matcher, distances, inf)
    d_unmatching = torch.where(not_matcher, distances, inf)
    dp = torch.min(d_matching, dim=-1, keepdim=True)
    dm = torch.min(d_unmatching, dim=-1, keepdim=True)
    if with_indices:
        return dp, dm
    return dp.values, dm.values


def one_class_classifier_triplet_loss(distances, target_labels, prototype_labels, theta_boundary):
    """ triplet loss for OneClassClassifier """
    zero = torch.tensor(0).type(torch.float)
    diff_to_thresh = torch.subtract(distances, theta_boundary)
    dp, dm = _get_dp_dm(diff_to_thresh, target_labels, prototype_labels, fill_value=0)
    mu = dp - dm
    mu = torch.add(mu, theta_boundary)
    mu = torch.where(mu > 0, mu, zero)  # + margin
    loss = torch.min(mu, dim=-1).values
    return loss


def _test_one_class_classifier_triplet_loss(distances, target_labels, prototype_labels, theta_boundary):
    """ triplet loss for OneClassClassifier """
    zero = torch.tensor(0).type(torch.float)
    diff_to_thresh = torch.subtract(distances, theta_boundary)
    dp, dm = _get_dp_dm(diff_to_thresh, target_labels, prototype_labels, fill_value=0)
    #dp, dm = _get_dp_dm(distances, target_labels, prototype_labels, fill_value=0)
    #mu = dp - dm # noch mal schauen
    dap = (dp - dm) / (dp + dm)
    dan = (dp - dm) / (dp + dm)
    mu = dap - dan
    #mu = torch.subtract(mu, theta_boundary)
    mu = torch.add(mu, theta_boundary)
    mu = torch.where(mu > 0, mu, zero)  # + margin
    print(mu)
    loss = torch.min(mu, dim=-1).values
    #print(loss)
    return loss


def _get_dop_in_diopf(diff_to_thresh, matcher, not_matcher, device='cpu'):
    zero = torch.tensor(0).type(torch.float).to(device)
    d_inner_pn = torch.where(diff_to_thresh < zero, diff_to_thresh, zero)
    d_inner_p = torch.where(matcher, d_inner_pn, zero)
    d_inner_n = torch.where(not_matcher, d_inner_pn, zero)
    d_outer_pn = torch.where(diff_to_thresh >= zero, diff_to_thresh, zero)
    d_outer_p = torch.where(matcher, d_outer_pn, zero)
    _zeros = torch.full(d_outer_p.shape, 0).type(torch.float).to(device)
    d_outer_p_free = torch.where(torch.min(d_inner_p, dim=1).values < zero, _zeros.T, d_outer_p.T).T
    d_op_in = torch.add(d_outer_p_free, d_inner_n)
    d_iopf = torch.add(d_outer_p_free, d_inner_p)
    return d_op_in, d_iopf


def _get_dop_din(diff_to_thresh, matcher, not_matcher):
    zero = torch.tensor(0).type(torch.float)
    d_inner_pn = torch.where(diff_to_thresh < zero, diff_to_thresh, zero)
    d_inner_n = torch.where(not_matcher, d_inner_pn, zero)
    d_outer_pn = torch.where(diff_to_thresh >= zero, diff_to_thresh, zero)
    d_outer_p = torch.where(matcher, d_outer_pn, zero)
    return d_outer_p, d_inner_n

    
    
def _backup_one_class_classifier_loss(distances, target_labels, prototype_labels, theta_boundary):
    """ OneClassClassifier loss function """
    zero = torch.tensor(0).type(torch.float)
    matcher = _get_matcher(target_labels, prototype_labels)
    not_matcher = torch.bitwise_not(matcher)

    # Optimizing False Positives and Negatives
    diff_to_thresh = torch.subtract(distances, theta_boundary)
    d_op_in, d_iopf = _get_dop_in_diopf(diff_to_thresh, matcher, not_matcher)
    muf = d_op_in * torch.pow(-1., torch.LongTensor(not_matcher.type(torch.long)))
    #muf += matcher * 100
    #muf = torch.min(muf, dim=-1, keepdims=True).values
    muf = torch.sum(muf, dim=-1, keepdims=True)

    # Optimizing Margin (theta_boundary)
    #d_unmatching = torch.where(not_matcher, distances, zero)
    #dp_max = torch.max(d_iopf, dim=-1, keepdim=True).values
    #dn_min = torch.min(d_unmatching, dim=-1, keepdim=True).values
    #mut = diff_to_thresh - torch.divide(torch.add(dp_max, dn_min),2)
    #mut = torch.min(mut, dim=-1, keepdims=True).values
    #mut = torch.sum(mut, dim=-1, keepdims=True) # führt dazu, das sich die Prototypen übereinander legen, auch ein Weglassen von 'mut'

    # Minimizing distances to True Positives (similar to penalty term)
    d_matching_zero = torch.where(matcher, distances, zero)
    mud = torch.min(d_matching_zero, dim=-1, keepdims=True).values
    #mud = torch.min(d_iopf, dim=-1, keepdims=True).values
    #mud = torch.mean(d_iopf, dim=-1, keepdims=True)
    #mud = torch.mean(d_matching_zero, dim=-1, keepdims=True)
 
    return muf + mud


def one_class_classifier_loss(distances, target_labels, prototype_labels, theta_boundary, device='cpu'):
    """ OneClassClassifier loss function """
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    zero = torch.tensor(0).type(torch.float).to(device)
    inf = torch.tensor(float('inf')).type(torch.float).to(device)
    matcher = _get_matcher(target_labels, prototype_labels, device=device)
    not_matcher = torch.bitwise_not(matcher)

    # Optimizing False Positives and Negatives
    diff_to_thresh = torch.subtract(distances, theta_boundary)
    d_op_in, d_iopf =_get_dop_in_diopf(diff_to_thresh, matcher, not_matcher, device=device)
    minus = torch.tensor(-1).type(torch.float).to(device)
    muf = d_op_in * torch.pow(minus, not_matcher.type(torch.long))
    muf = torch.sum(muf, dim=-1, keepdims=True)
    
    # Minimizing distances to True Positives (similar to penalty term)
    d_matching_zero = torch.where(matcher, distances, zero)
    d_matching_inf = torch.where(matcher, distances, inf)
    # when having multiple classes some distances > 0 get 0 bc of protos with false label
    # we check this with the sum of distances for all protos
    dsums = torch.sum(d_matching_zero, dim=-1, keepdims=True)
    dzmins = torch.min(d_matching_zero, dim=-1, keepdims=True).values
    dimins = torch.min(d_matching_inf, dim=-1, keepdims=True).values
    mud = torch.where(dsums != 0., dimins, dzmins)

    alpha = torch.tensor(0.5).type(torch.float).to(device)
    opalpha = torch.tensor(1 - alpha).type(torch.float).to(device)

    mu = alpha * mud + opalpha * muf

    return mu.mean()


def occ_loss(distances, target_labels, prototype_labels, theta_boundary):
    """ OneClassClassifier loss function """
    zero = torch.tensor(0).type(torch.float)
    matcher = _get_matcher(target_labels, prototype_labels)
    not_matcher = torch.bitwise_not(matcher)

    # Optimizing False Positives and Negatives
    diff_to_thresh = torch.subtract(distances, theta_boundary)
    d_op, d_in = _get_dop_din(diff_to_thresh, matcher, not_matcher)
    dp, dm = _get_dp_dm(diff_to_thresh, target_labels, prototype_labels, fill_value=0)
    #muf = d_op_in * torch.pow(-1., torch.LongTensor(not_matcher.type(torch.long)))
    """
    loss_pen = alpha * dp - (1 - alpha) * dm
    loss_pen = alpha * dp - 1 * dm + alpha * dm
    loss_pen = alpha * (dp + dm) - dm
    """
    loss_pen = 10
    #alpha = (loss_pen + dm) / (dp + dm)
    #alpha = (loss_pen + d_in) / (d_op + d_in)
    alpha = (loss_pen + d_in) / (dp + d_in)
    alpha = torch.where(alpha <= 0., torch.tensor(0).type(torch.float), alpha)
    alpha = torch.where(alpha >= 0.5, torch.tensor(0.5).type(torch.float), alpha)
    alpha = 0.2
    #muf = alpha * dp - (1 - alpha) * dm
    #muf = alpha * d_op - (1 - alpha) * d_in
    muf = alpha * dp - (1 - alpha) * d_in
    #muf = alpha + d_op - (1 - alpha + dm)
    muf = torch.sum(muf, dim=-1, keepdims=True)

    # Minimizing distances to True Positives (similar to penalty term)
    d_matching_zero = torch.where(matcher, distances, zero)
    mud = torch.min(d_matching_zero, dim=-1, keepdims=True).values
    #mud = torch.min(d_iopf, dim=-1, keepdims=True).values

    return muf + mud


def _backup_occ_loss(distances, target_labels, prototype_labels, theta_boundary):
    """ OneClassClassifier loss function """
    zero = torch.tensor(0).type(torch.float)
    matcher = _get_matcher(target_labels, prototype_labels)
    not_matcher = torch.bitwise_not(matcher)

    # Optimizing False Positives and Negatives
    diff_to_thresh = torch.subtract(distances, theta_boundary)
    d_op, d_in = _get_dop_din(diff_to_thresh, matcher, not_matcher)
    dp, dm = _get_dp_dm(diff_to_thresh, target_labels, prototype_labels, fill_value=0)
    #muf = d_op_in * torch.pow(-1., torch.LongTensor(not_matcher.type(torch.long)))
    """
    loss_pen = alpha * dp - (1 - alpha) * dm
    loss_pen = alpha * dp - 1 * dm + alpha * dm
    loss_pen = alpha * (dp + dm) - dm
    """
    #loss_pen = 100
    #alpha = (loss_pen + dm) / (dp + dm)
    #alpha = (loss_pen + d_in) / (dp + d_in)
    #alpha = torch.where(alpha <= 0., torch.tensor(0).type(torch.float), alpha)
    #alpha = torch.where(alpha >= 1., torch.tensor(1).type(torch.float), alpha)
    alpha = 0.99 
    muf = alpha * d_op - (1 - alpha) * d_in
    muf = torch.sum(muf, dim=-1, keepdims=True)

    return muf

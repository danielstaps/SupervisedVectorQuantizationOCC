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

    """
    Heaviside -> RELU oder SIGMOID
    Funktion umschreiben
    """

    # Optimizing False Positives and Negatives
    #winning_indices = torch.min(distances, dim=1).indices
    #diff_to_thresh = torch.subtract(distances[winning_indices], theta_boundary)
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

    #mu = alpha * mud + opalpha * muf
    mu = muf

    return mu.mean()


def occ_mitRonny(distances, target_labels, prototype_labels, theta_boundary, device='cpu'):
    """ OneClassClassifier loss function """

    print("theta",theta_boundary)
    print(device)

    zero = torch.tensor(0).type(torch.float).to(device)
    inf = torch.tensor(float('inf')).type(torch.float).to(device)
    matcher = _get_matcher(target_labels, prototype_labels, device=device)
    not_matcher = torch.bitwise_not(matcher)

    # Optimizing False Positives and Negatives
    winning_indices = torch.min(distances, dim=1).indices # Liste an Winning protos
    """
    winning_indices = torch.expand_dims(winning_indices, axis=0)
    print(winning_indices.shape)
    d_tilde = torch.subtract(distances, theta_boundary)
    print(d_tilde.shape)
    d_tilde = d_tilde[:,winning_indices]
    """
    d_tilde = torch.subtract(distances, theta_boundary)
    d_tilde = d_tilde.gather(1, winning_indices.unsqueeze(1)).squeeze()
    not_matcher = not_matcher.gather(1, winning_indices.unsqueeze(1)).squeeze()
    #print(d_tilde)
    # d_tilde = torch.subtract(distances, theta_boundary)
    minus = torch.tensor(-1).type(torch.float).to(device)
    #muf = d_tilde * torch.pow(minus, not_matcher.type(torch.long))
    muf = d_tilde * ((-1) ** not_matcher.type(torch.long))
    # RELU
    muf = torch.relu(muf)
    #d_op_in, _ =_get_dop_in_diopf(d_tilde, matcher, not_matcher, device=device)
    # SIGMOID
    #muf = torch.sigmoid(muf)
    #print(muf, muf.shape)
    #print(muf)
    #muf = torch.sum(muf, dim=-1, keepdims=True)
    #print(muf)

    # Minimizing distances to True Positives (similar to penalty term)
    d_matching_zero = torch.where(matcher, distances, zero)
    d_matching_inf = torch.where(matcher, distances, inf)
    # when having multiple classes some distances > 0 get 0 bc of protos with false label
    # we check this with the sum of distances for all protos
    dsums = torch.sum(d_matching_zero, dim=-1, keepdims=True)
    dzmins = torch.min(d_matching_zero, dim=-1, keepdims=True).values
    dimins = torch.min(d_matching_inf, dim=-1, keepdims=True).values
    mud = torch.where(dsums != 0., dimins, dzmins)

    mu = mud + muf

    return mu.mean()



""" implementation of student-t distribution """
def studentT(distances, theta_boundary):
    torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732

    #print("theta",theta_boundary, theta_boundary.shape)
    prefactor = 1 / (torch.pi * theta_boundary)
    #print("prefa",prefactor, prefactor.shape)

    winning_indices = torch.min(distances, dim=1).indices # list of winning prototypes
    #print("distances",distances)
    distribution = 1 / (1 + (distances / (theta_boundary ** 2)))

    studentT = prefactor * distribution
    #studentT = distribution
    studentT = studentT.gather(1, winning_indices.unsqueeze(1)).squeeze()

    return studentT


def error_type_determination(distances, theta_boundary, target_labels, prototype_labels):
    matcher = _get_matcher(target_labels, prototype_labels)
    not_matcher = torch.bitwise_not(matcher)

    winning_indices = torch.min(distances, dim=1).indices # list of winning prototypes
    d_tilde = torch.subtract(distances, theta_boundary)

    #print("d_tilde",d_tilde[:10])
    #print("matcher",matcher[:10])

    is_in_bound = d_tilde < 0
    is_out_of_bound = d_tilde >= 0
    case1 = torch.logical_and(is_out_of_bound, matcher)
    #case1 = torch.where(torch.logical_and(is_out_of_bound, matcher.squeeze()), -1, 0)
    case2 = torch.logical_and(is_in_bound, not_matcher)
    #print(case1 == case2)
    
    #fF = torch.add(case1, case2)
    #fF = fF.gather(1, winning_indices.unsqueeze(1)).squeeze()
    #print("fF",fF)
    FN = case1.gather(1, winning_indices.unsqueeze(1)).squeeze()
    FP = case2.gather(1, winning_indices.unsqueeze(1)).squeeze()

    return FN, FP


def occ_studentT_loss(distances, target_labels, prototype_labels, theta_boundary, device='cpu'):
    """
    OneClassClassifier loss function implemented with Student-t distribution
    """

    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    # get probabilty from distribution
    prob = studentT(distances, theta_boundary)
    #print("\nprobability sT:",prob)

    # filter FP, FN
    FN, FP = error_type_determination(distances, theta_boundary, target_labels, prototype_labels) 
    #print("\nerrortypedeter:",fF)

    # calc loss
    #FPloss = torch.masked_select(prob, FP)
    #FNloss = 1 - torch.masked_select(prob, FN)
    #FPloss = torch.where(FP, prob.float(), 0.)
    #FNloss = 1. - torch.where(FN, prob.float(), 0.)
    FPloss = (FP * prob)
    FNloss = 1 - (FN * prob)
    #print(FPloss.mean(), FNloss.mean())
    #print(torch.cat([FPloss, FNloss]))

    loss = (FPloss + FNloss).mean()
    #loss = torch.cat([FPloss, FNloss]).mean()
    #loss = (fF * prob).mean()
    #print("loss",loss)
    #print(fF, prob)

    return loss
    


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

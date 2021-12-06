import torch



def _get_matcher(targets, labels, device='cpu'):
    """Returns a boolean tensor."""
    #matcher = torch.eq(targets.unsqueeze(dim=1), labels)
    targets_resized = targets.unsqueeze(dim=1)
    matcher = torch.eq(targets_resized.to(device), labels.to(device))
    if labels.ndim == 2:
        # if the labels are one-hot vectors
        num_classes = targets.size()[1]
        matcher = torch.eq(torch.sum(matcher, dim=-1), num_classes)
    return matcher


def error_type_determination(distances, theta_boundary, target_labels, prototype_labels, device='cpu', pass_errors=False):
    matcher = _get_matcher(target_labels, prototype_labels, device=device)
    not_matcher = torch.bitwise_not(matcher)

    d_tilde = torch.subtract(distances.to(device), theta_boundary.to(device))
    #print("d_tilde",d_tilde[:10])
    #print("matcher",matcher[:10])

    is_in_bound = d_tilde < 0
    is_out_of_bound = d_tilde >= 0

    TP = torch.logical_and(is_in_bound, matcher)
    FN = torch.logical_and(is_out_of_bound, matcher)
    TN = torch.logical_and(is_out_of_bound, not_matcher)
    FP = torch.logical_and(is_in_bound, not_matcher)
    if pass_errors:
        return TP, TN, FP, FN

    #fF = torch.add(case1, case2)
    #fF = fF.gather(1, winning_indices.unsqueeze(1)).squeeze()
    #print("fF",fF)

    winning_indices = torch.min(distances, dim=1).indices.to(device) # list of winning prototypes
    TP = TP.gather(1, winning_indices.unsqueeze(1)).squeeze()
    TN = TN.gather(1, winning_indices.unsqueeze(1)).squeeze()
    FP = FP.gather(1, winning_indices.unsqueeze(1)).squeeze()
    FN = FN.gather(1, winning_indices.unsqueeze(1)).squeeze()

    return TP, TN, FP, FN


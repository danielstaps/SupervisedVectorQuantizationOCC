"""ProtoTorch competition functions."""

import torch


def wtac_thresh(distances: torch.Tensor, labels: torch.LongTensor,
                theta_boundary: torch.Tensor) -> (torch.LongTensor):
    """ Used for OneClassClassifier.
    Calculates if distance is in between the Voronoi-cell of prototype or not. Voronoi-cell is defined by >theta_boundary<. (like a radius) """
    #in_boundary = (theta_boundary - distances)
    #winning_indices = torch.min(in_boundary, dim=1).indices
    #if torch.cuda.is_available():
    #    device = 'cuda:0'
    #else:
    device = 'cpu'
    distances = distances.to(device)
    theta_boundary = theta_boundary.to(device)
    winning_indices = torch.min(distances, dim=1).indices
    labels = labels.to(device)
    winning_labels = labels[winning_indices].squeeze()
    in_boundary = (theta_boundary - distances)
    in_boundary = in_boundary.gather(1, winning_indices.unsqueeze(1)).squeeze()
    zero = torch.tensor(0.).type(torch.float).to(device)
    winning_labels = torch.where(in_boundary > zero, winning_labels,
                                 torch.max(labels) +
                                 1)  # '-1' -> 'garbage class'
    return winning_labels

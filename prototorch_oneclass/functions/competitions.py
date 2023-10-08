"""ProtoTorch competition functions."""

import torch


def wtac_thresh(
    distances: torch.Tensor,
    labels: torch.Tensor,
    theta_boundary: torch.Tensor,
) -> torch.Tensor:
    """ Used for OneClassClassifier.
    Calculates if distance is in between the Voronoi-cell of prototype or not. Voronoi-cell is defined by >theta_boundary<. (like a radius) """

    # so wie in WTAC
    winning_indices = torch.min(distances, dim=1).indices
    winning_labels = labels[winning_indices].squeeze()

    in_boundary = (theta_boundary - distances)
    # d > 0: in boundary, d < 0: out of boundary
    in_boundary = in_boundary.gather(1, winning_indices.unsqueeze(1)).squeeze()

    winning_labels = torch.where(
        in_boundary > 0.0,
        winning_labels,
        torch.max(labels) + 1,
    )

    return winning_labels


class WTAC_Thresh(torch.nn.Module):
    """Winner-Takes-All-Competition Layer.
    Thin wrapper over the `wtac` function.
    """

    def __init__(self, theta_boundary):
        super().__init__()
        self.theta_boundary = theta_boundary

    def forward(self, distances, labels):  # pylint: disable=no-self-use
        return wtac_thresh(distances, labels, self.theta_boundary)

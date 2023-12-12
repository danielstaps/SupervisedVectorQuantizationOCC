import math

import torch


def sigmoid(boundary_distance, sigma=0.1):
    """
    implementation of sigmoid
    """
    # sigma = 0.1
    exponent = -boundary_distance / sigma
    mask = torch.where(exponent >= 20, 0.0, 1.0)  # for numerical stablity
    exponent *= mask
    sig = 1 / (1 + torch.exp(exponent))
    sig *= mask
    return sig


def uniform_fct(squared_distances, theta_boundary):
    """
    Uniform distribution

    squared_distances: squared distances to the prototypes
    """
    uniform = torch.ones(squared_distances.size()[0], theta_boundary.size()[0])
    return uniform


def studentT_fct(squared_distances, theta_boundary):
    """
    Student-t distribution for a single degree of freedom

    squared_distances: squared distances to the prototypes
    """
    prefactor = 1 / (math.pi * theta_boundary)
    distribution = 1 / (1 + (squared_distances / (theta_boundary**2)))

    studentT = prefactor * distribution
    return studentT


def gaussian_fct(squared_distances, theta_boundary):
    """
    squared_distances: squared distances to the prototypes
    """
    prefactor = 1 / (theta_boundary * math.sqrt(2 * math.pi))

    exponent = -(1 / 2) * (squared_distances / theta_boundary) ** 2
    distribution = torch.exp(exponent)

    gauss = prefactor * distribution
    return gauss


DISTRIBUTIONS = {
    "uniform": uniform_fct,
    "studentT": studentT_fct,
    "gauss": gaussian_fct,
}


def get_probabilities(squared_distances, theta_boundary, distribution, zero_mean):
    if distribution not in DISTRIBUTIONS:
        raise ValueError(
            f"Unknown distribution {distribution} for distribution_handler, choose from {list(DISTRIBUTIONS.keys())}"
        )

    probs = DISTRIBUTIONS[distribution](squared_distances, theta_boundary)

    # normalize
    norm_scalar = DISTRIBUTIONS[distribution](
        zero_mean,
        theta_boundary,
    )
    probs = probs / norm_scalar

    return probs

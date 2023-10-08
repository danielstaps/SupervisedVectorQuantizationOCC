import numpy as np
import torch
from scipy.spatial import distance_matrix


def make_S(
    n_samples,
    center=None,
    radius=0.5,
    noise=0.3,
    ratio=0.5,
):
    # god bless polar coordinates
    # create S-shaped class

    if center is None:
        center = radius

    # first 'circle' centered at center with radius r
    t1 = np.random.uniform(0., 3 * np.pi / 2, int(n_samples / 2))
    x1 = radius * np.cos(t1)
    y1 = radius * np.sin(t1) + center

    # second 'circle'
    t2 = np.random.uniform(np.pi / 2, 2 * np.pi, int(n_samples / 2))
    x2 = -radius * np.cos(t2)
    y2 = radius * np.sin(t2) - center

    # concat the above circles to create S-shape
    x = np.concatenate((x1, x2))
    y = np.concatenate((y1, y2))

    # salt the coordinates
    x_with_noise = x + noise * np.random.uniform(-1, 1, n_samples)
    y_with_noise = y + noise * np.random.uniform(-1, 1, n_samples)

    real_coords = np.dstack((x, y)).squeeze()
    salted_coords = np.dstack((x_with_noise, y_with_noise)).squeeze()

    # calculate the distances between any two points of the unsalted and salted coordinates
    # pretty costly
    # TODO: find alternative for defining non-target class
    distances = distance_matrix(real_coords, salted_coords)
    minimus = np.min(distances, axis=0)
    labels = np.where(minimus < noise * ratio, 0, 1)

    dataset = salted_coords

    return dataset, labels


class S_Shape(torch.utils.data.TensorDataset):
    def __init__(self,
                 n_samples: int = 800,
                 center: float = 0.5,
                 radius: float = 0.5,
                 noise: float = 0.3,
                 ratio: float = 0.5,
                 **kwargs):
        x, y = make_S(n_samples, radius, noise, ratio)
        self.data = torch.squeeze(torch.Tensor(x))
        self.target = torch.squeeze(torch.LongTensor(y))
        super().__init__(self.data, self.target)

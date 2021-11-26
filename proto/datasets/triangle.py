""" Datasets for OneClassClassifier inspired by the PalauFlag """

import matplotlib.pyplot as plt

import numpy as np
import torch

np.random.seed(42)


def make_flag(num_samples=300, dimensions=2, num_classes=1):
    """ Generates a variation of the PalauFlag """

    num_classes = 3

    x,y  = [],[]
    for i in range(num_classes):
        x_ = np.random.uniform(low=-0.1, high=[1.1, 0.2], size=(num_samples//num_classes, dimensions))
        phi = - i * 1/3 * np.pi
        rot = np.asarray([[np.cos(phi),-np.sin(phi)],[np.sin(phi),np.cos(phi)]])
        x_ = np.matmul(x_, rot)
        if i == 2:
            x_ = x_ + np.asarray([1.0,0.0]).T
        x.append(x_)
        y.append(np.full(num_samples//num_classes, i))
    x = np.concatenate(x, axis=0)
    y = np.concatenate(y, axis=0)
    print(x.shape, y.shape)

    return x, y


class Triangle(torch.utils.data.TensorDataset):
    """ Flag dataset for OOC.

    This datasets consists of n blobs of different classes and a background garbage noise.

    .. list-table:: Flag
        :header-rows: 1

        * - dimensions
          - classes
          - training size
          - validation size
          - test size
        * - 2
          - n
          - num_samples
          - 0
          - 0

    :param num_samples: number of random samples
    :param noise: noise added to the spirals
    """
    def __init__(self, num_samples: int = 500, dimensions: int = 2, num_classes: int = 1):
        x, y = make_flag(num_samples, dimensions, num_classes)
        self.data = torch.Tensor(x)
        self.target = torch.LongTensor(y)
        super().__init__(self.data, self.target)


if __name__ == '__main__':
    Triangle(num_samples=1000, dimensions=2)

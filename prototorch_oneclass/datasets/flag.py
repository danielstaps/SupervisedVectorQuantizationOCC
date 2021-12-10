""" Datasets for OneClassClassifier inspired by the PalauFlag """

import numpy as np
import torch

np.random.seed(42)


def make_flag(num_samples=300,
              dimensions=2,
              num_classes=1,
              blobs_per_class=1,
              elipses=True):
    """ Generates a variation of the PalauFlag """
    x = np.random.uniform(low=0.0, high=1.0, size=(num_samples, dimensions))
    y = np.full(num_samples, num_classes, dtype=int)

    test = 1

    blob_mean = np.random.uniform(low=0.1,
                                  high=0.9,
                                  size=(num_classes * blobs_per_class,
                                        dimensions))
    #blob_mean = np.random.uniform(low=0.5, high=0.5, size=(num_classes*blobs_per_class, dimensions))
    #blob_mean[:,0] = 0.2
    blob_bord = np.random.uniform(low=0.05 * dimensions * test,
                                  high=0.15 * dimensions * test,
                                  size=(num_classes * blobs_per_class,
                                        dimensions))

    for target in range(num_classes):
        for blob in range(blobs_per_class):
            obj = target + (blob * num_classes)
            mean, bord = blob_mean[obj], blob_bord[obj]
            x_shifted = np.subtract(x, mean)
            x_shf_pow = np.power(x_shifted, 2)
            bbord_pow = np.power(bord, 2)
            distances = np.sum(np.divide(x_shf_pow, bbord_pow), axis=1)
            in_border = np.where(distances < 1.)[0]
            y[in_border] = target

    print(x.shape, y.shape)

    return x, y


class Flag(torch.utils.data.TensorDataset):
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
    def __init__(self,
                 num_samples: int = 500,
                 dimensions: int = 2,
                 num_classes: int = 1,
                 blobs_per_class: int = 1,
                 elipses: bool = True):
        x, y = make_flag(num_samples, dimensions, num_classes, blobs_per_class,
                         elipses)
        self.data = torch.Tensor(x)
        self.target = torch.LongTensor(y)
        super().__init__(self.data, self.target)

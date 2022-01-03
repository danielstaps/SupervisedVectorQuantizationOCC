import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import make_blobs, make_circles, make_moons

np.random.seed(42)


def create_shapes(num_samples=300,
                  num_shapes=1,
                  num_classes=1,
                  outliers=0.1,
                  **kwargs):

    keys = ['centers', 'cluster_std', 'random_state', 'noise', 'factor']
    for key in keys:
        if key not in kwargs.keys() and key != 'factor':
            kwargs[key] = None
        if key not in kwargs.keys() and key == 'factor':
            kwargs[key] = 0.8

    #outlier = int(outliers*num_samples)
    #num_samples = num_samples - outlier

    types = {
        'blobs':
        make_blobs(n_samples=num_samples,
                   centers=kwargs['centers'],
                   cluster_std=kwargs['cluster_std'],
                   random_state=kwargs['random_state']),
        'circle':
        make_circles(n_samples=num_samples,
                     noise=kwargs['noise'],
                     random_state=kwargs['random_state'],
                     factor=kwargs['factor']),
        'moon':
        make_moons(n_samples=num_samples,
                   noise=kwargs['noise'],
                   random_state=kwargs['random_state'])
    }

    if kwargs['kind'] is None:
        kwargs['kind'] = 'moon'
    if not any(map(lambda shape: shape in kwargs['kind'], types.keys())):
        msg = 'kind must be in [{}, {}, {}]'.format(*types.keys())
        raise ValueError(msg)

    dataset, targets = [], []
    # TODO: extend for multiple shapes per class
    for i in range(num_classes):
        for j in range(num_shapes):
            dataset.append(types[kwargs['kind']])

    datasets, targets = zip(*dataset)
    return datasets, targets


class Shapes(torch.utils.data.TensorDataset):
    def __init__(self,
                 num_samples: int = 300,
                 num_shapes: int = 1,
                 num_classes: int = 1,
                 outliers=0.1,
                 **kwargs):
        x, y = create_shapes(num_samples, num_shapes, num_classes, outliers,
                             **kwargs)
        self.data = torch.squeeze(torch.Tensor(x))
        print(self.data.shape)
        self.target = torch.squeeze(torch.LongTensor(y))
        super().__init__(self.data, self.target)

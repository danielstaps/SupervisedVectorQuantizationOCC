import numpy as np
import torch
from sklearn.datasets import make_blobs, make_circles, make_moons

np.random.seed(42)


def create_shapes(num_samples=300,
                  num_shapes=1,
                  num_classes=1,
                  outliers=0.0,
                  **kwargs):

    keys = ['centers', 'cluster_std', 'random_state', 'noise', 'factor']
    for key in keys:
        if key not in kwargs.keys() and key != 'factor':
            kwargs[key] = None
        if key not in kwargs.keys() and key == 'factor':
            kwargs[key] = 0.8

    outlier = int(outliers * num_samples)
    num_samples = num_samples - outlier

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
    for j in range(num_shapes):
        dataset.append(types[kwargs['kind']])

    datasets, targets = zip(*dataset)
    datasets = np.asarray(datasets)
    targets = np.where(targets[0] <= num_classes, targets[0], num_classes)
    return datasets, targets


class Shapes(torch.utils.data.TensorDataset):
    def __init__(self,
                 num_samples: int = 300,
                 num_shapes: int = 1,
                 num_classes: int = 1,
                 outliers=0.0,
                 **kwargs):
        x, y = create_shapes(num_samples, num_shapes, num_classes, outliers,
                             **kwargs)
        self.data = torch.squeeze(torch.Tensor(x))
        print(self.data.shape)
        self.target = torch.squeeze(torch.LongTensor(y))
        super().__init__(self.data, self.target)


if __name__ == '__main__':
    # Configuration
    kind = 'blobs'
    num_samples = 1000
    num_shapes = 1
    num_classes = 2
    outliers = 0.1
    center = [[-0.1, 0.15], [0.1, 0.1], [-0.3, 0.1]]
    cluster_std = [0.08, 0.05, 0.02]
    random_state = 421
    noise = 0.15
    factor = 0.8
    prototypes_per_class = 1

    kwargs = {
        'kind': kind,
        'centers': center,
        'cluster_std': cluster_std,
        'random_state': random_state,
        'noise': noise,
        'factor': factor,
    }

    # Dataset
    train_ds = Shapes(
        num_samples=num_samples,
        num_shapes=num_shapes,
        num_classes=num_classes,
        outliers=outliers,
        **kwargs,
    )

    print(train_ds.data, train_ds.target)

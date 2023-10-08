""" Datasets for OneClassClassifier inspired by the PalauFlag """

import matplotlib.pyplot as plt
import numpy as np
import torch

np.random.seed(42)


def make_polygon(num_samples=300, dimensions=2, num_classes=3, thickness=0.1):

    assert num_classes >= 3, "num_classes must be >=3"

    x, y = [], []
    x_len = np.tan(np.pi / num_classes)
    for i in range(num_classes):
        phi = (i / num_classes) * 2 * np.pi
        x_ = np.random.uniform(
            low=[-1 - thickness, -x_len - thickness],
            high=[-1 + thickness, x_len + thickness],
            size=(num_samples // num_classes, dimensions),
        )
        rot = np.asarray([[np.cos(phi), -np.sin(phi)],
                          [np.sin(phi), np.cos(phi)]])
        x_ = np.matmul(x_, rot)
        x.append(x_)
        y.append(np.full(num_samples // num_classes, i))

    x = np.concatenate(x, axis=0)
    y = np.concatenate(y, axis=0)

    return x, y


class Polygon(torch.utils.data.TensorDataset):

    def __init__(
        self,
        num_samples: int = 500,
        dimensions: int = 2,
        num_classes: int = 3,
        thickness: float = 0.1,
    ):
        x, y = make_polygon(num_samples, dimensions, num_classes, thickness)
        self.data = torch.Tensor(x)
        self.target = torch.LongTensor(y)
        super().__init__(self.data, self.target)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    for i in range(2):
        for j in range(4):
            num_classes = j + 3
            x, y = make_polygon(
                num_samples=1000,
                dimensions=2,
                num_classes=num_classes,
                thickness=0.1 + 0.5 * i,
            )
            plt.subplot(2, 4, (i * 4) + j + 1)
            plt.axis('equal')
            plt.scatter(x[:, 0], x[:, 1], c=y)
    plt.show()

""" Datasets for OneClassClassifier inspired by the PalauFlag """
import os
import pandas as pd
import numpy as np
import torch

np.random.seed(42)


def prepare_quad(data, feature):
    if feature == "bow_vec":
        vec = data["bow_vec"].values.tolist()
    elif feature == "nat_vec":
        vec = data["nat_vec"].values.tolist()
    elif feature == "mif_vec":
        vec = data["mif_vec"].values.tolist()
    elif feature == "rmif_vec":
        vec = data["rmif_vec"].values.tolist()
    x = []
    for item in vec:
        row = np.array([float(i) for i in item.replace("[","").replace("]","").replace(" ","").split(",")])
        x.append(row)
    labels = data["loopTopology"].values.tolist()
    y = []
    for item in labels:
        if item == "1a":
            y.append(0)
        else:
            y.append(1)
    x = np.array(x)
    y = np.array(y)
    return x, y

def make_quad(feature=None):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data = pd.read_csv(f"{dir_path}/data_roh/noise_mixture_data.csv")
    x, y = prepare_quad(data, feature)

    print(x.shape, y.shape)
    return x, y


class Quad():
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
    def __init__(self, feature: str = "bow_vec",):
        x, y = make_quad(feature)
        #self.data = torch.Tensor(x)
        #self.target = torch.LongTensor(y)
        self.data = x
        self.target = y
        #super().__init__(self.data, self.target)


if __name__ == '__main__':
    Quad(feature="bow_vec")

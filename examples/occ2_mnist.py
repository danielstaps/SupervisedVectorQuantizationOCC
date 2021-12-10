"""GLVQ example using the spiral dataset."""

import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np
import prototorch as pt
import pytorch_lightning as pl
import torch
from img_preproc import give_data_back
#from proto.datasets.flag import Flag
from proto.oneclass import OneClassGLVQ, OneClassGMLVQ, OneClassLGMLVQ
from prototorch.datasets import NumpyDataset
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix
from torchvision.datasets import MNIST

CUDA = True

if __name__ == "__main__":
    # Command-line arguments
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # Dataset
    num_classes = 1
    """
    mnist = load_digits()
    print(mnist.keys())

    print(mnist['data'].shape, mnist['target'].shape)
    x = mnist['data']/np.amax(mnist['data'])
    y = mnist['target']
    y = np.where(y == 4, 0, 1)
    print(y)

    train_ds = NumpyDataset(x, y)

    # Dataloaders
    train_loader = torch.utils.data.DataLoader(train_ds,
                                               num_workers=0,
                                               batch_size=train_ds.data.shape[0],
                                               )
    """

    (x_train, y_train), (x_test, y_test) = give_data_back()

    train_ds = NumpyDataset(x_train, y_train)
    # Dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        num_workers=4,
        #batch_size=train_ds.data.shape[0]//2000,
        batch_size=train_ds.data.shape[0] // 10
        #batch_size=1000,
        #batch_size=100
    )

    test_ds = NumpyDataset(x_test, y_test)
    # Dataloaders
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        num_workers=4,
        batch_size=test_ds.data.shape[0],
    )

    # Hyperparameters
    prototypes_per_class = 3
    hparams = dict(
        distribution=(num_classes, prototypes_per_class),
        input_dim=x_train.shape[1],
        latent_dim=10,
        #transfer_function="sigmoid_beta",
        #transfer_beta=10.0,
        proto_lr=0.01,
        bb_lr=0.01,
        #lr=0.01,
    )

    # Initialize the model
    model = OneClassGMLVQ(
        hparams,
        optimizer=torch.optim.Adam,
        #prototypes_initializer=pt.core.SMCI(train_ds),
        prototypes_initializer=pt.core.SSCI(train_ds, noise=5e-2),
    )

    # Callbacks
    vis = pt.models.VisGMLVQ2D(train_ds, show_last_only=False, block=False)
    pruning = pt.models.PruneLoserPrototypes(
        threshold=0.01,
        idle_epochs=1,
        prune_quota_per_epoch=1,
        frequency=1,
        verbose=True,
    )

    # Setup trainer
    if CUDA:
        trainer = pl.Trainer.from_argparse_args(
            args,
            callbacks=[
                #vis,
                # pruning,
            ],
            terminate_on_nan=True,
            gpus='0')
    else:
        trainer = pl.Trainer.from_argparse_args(
            args,
            callbacks=[
                vis,
                # pruning,
            ],
            terminate_on_nan=True,
        )
    # Training loop
    trainer.fit(model, train_loader)

    # Testing
    trainer.test(model, test_dataloaders=test_loader)

    # Confusion matrix
    x_test = torch.Tensor(x_test)
    d = model.compute_distances(x_test)
    y_pred = model.predict_from_distances(d)

    print(confusion_matrix(y_test, y_pred.cpu().numpy()))

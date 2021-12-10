"""GLVQ example using the spiral dataset."""

import argparse

import numpy as np
import prototorch as pt
import pytorch_lightning as pl
import torch
from proto.datasets.flag import Flag
from proto.oneclass import OneClassGMLVQ
from pytorch_lightning.callbacks import LearningRateMonitor

CUDA = True

if __name__ == "__main__":
    # Command-line arguments
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # Dataset
    num_samples = 1000
    dimensions = 2
    num_classes = 1
    blobs_per_class = 1
    latent_dim = 2

    # Dataset
    #train_ds = pt.datasets.Spiral(num_samples=num_samples, noise=0.5)
    train_ds = Flag(num_samples, dimensions, num_classes, blobs_per_class)

    # Dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        num_workers=0,
        #batch_size=train_ds.data.shape[0],
        batch_size=num_samples,
    )

    # Hyperparameters
    prototypes_per_class = 1
    hparams = dict(
        input_dim=dimensions,
        latent_dim=latent_dim,
        distribution=(num_classes, prototypes_per_class),
        #transfer_function="sigmoid_beta",
        #transfer_beta=10.0,
        #lr=0.1,
        proto_lr=0.01,
        bb_lr=0.01,
    )

    # Initialize the model
    model = OneClassGMLVQ(
        hparams,
        optimizer=torch.optim.Adam,
        prototypes_initializer=pt.core.SMCI(train_ds),
        theta_initializer=train_ds.data[train_ds.target == 0]
        #prototypes_initializer=pt.core.SSCI(train_ds, noise=1e-2),
        #omega_initializer=pt.core.PCALTI(train_ds.data),
    )

    # Callbacks
    vis = pt.models.VisGLVQ2D(train_ds, show_last_only=False, block=False)
    pruning = pt.models.PruneLoserPrototypes(
        threshold=0.01,
        idle_epochs=1,
        prune_quota_per_epoch=1,
        frequency=1,
        verbose=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Setup trainer
    if CUDA:
        trainer = pl.Trainer.from_argparse_args(
            args,
            callbacks=[
                vis,
                lr_monitor,
                # pruning,
            ],
            terminate_on_nan=True,
            gpus='0')
    else:
        trainer = pl.Trainer.from_argparse_args(
            args,
            callbacks=[
                vis,
                lr_monitor,
                # pruning,
            ],
            terminate_on_nan=True,
        )
    # Training loop
    trainer.fit(model, train_loader)

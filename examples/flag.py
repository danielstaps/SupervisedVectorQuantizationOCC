"""GLVQ example using the spiral dataset."""
import argparse

import prototorch as pt
import pytorch_lightning as pl
import torch
# Prototorch One Class Classifier
from prototorch_oneclass import OneClassLGMLVQ
from prototorch_oneclass.datasets import Flag
from prototorch_oneclass.functions.callbacks import (SigmaCallback,
                                                     ThetaCallback)
from prototorch_oneclass.functions.losses import csi_soft_loss

# Configuration
num_classes = 2
num_samples = 1000
dimensions = 2
thickness = 0.4
prototypes_per_class = 1

if __name__ == "__main__":
    # Command-line arguments
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # Dataset
    train_ds = Flag(
        num_samples=num_samples,
        dimensions=dimensions,
        num_classes=num_classes,
    )

    # Dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        num_workers=0,
        batch_size=train_ds.data.shape[0],
    )

    # Hyperparameters
    hparams = dict(
        distribution=(num_classes, prototypes_per_class),
        input_dim=2,
        latent_dim=2,
        proto_lr=0.01,
        bb_lr=1.0,
    )

    # Initialize the model
    model = OneClassLGMLVQ(
        hparams,
        optimizer=torch.optim.Adam,
        prototypes_initializer=pt.core.SMCI(train_ds),
        theta_initializer=train_ds,
        loss=csi_soft_loss,
        theta_trainable=True,
    )

    # Callbacks
    vis = pt.models.VisGLVQ2D(train_ds, show_last_only=False, block=False)

    # Setup trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[
            vis,
            ThetaCallback(),
            SigmaCallback(),
        ],
        detect_anomaly=True,
    )

    # Training loop
    trainer.fit(model, train_loader)

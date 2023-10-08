"""GLVQ example using the spiral dataset."""
import argparse

import prototorch as pt
import pytorch_lightning as pl
import torch
# Prototorch One Class Classifier
from prototorch_oneclass import SVQ_OCC
from prototorch_oneclass.datasets import Flag
from prototorch_oneclass.functions.callbacks import (DynamicCallback,
                                                     ThetaCallback)
from prototorch_oneclass.functions.losses import (csi_soft_loss,
                                                  occ_entropy_loss,
                                                  brier_score)

# Configuration
num_classes = 1
num_samples = 1000
dimensions = 2
thickness = 0.4
prototypes_per_class = 7

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
        blobs_per_class=3,
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
        theta_lr=0.01,
        #bb_lr=0.0,
    )

    # Initialize the model
    model = SVQ_OCC(
        hparams,
        optimizer=torch.optim.Adam,
        prototypes_initializer=pt.core.SSCI(train_ds),
        theta_initializer=train_ds,
        loss=brier_score,
        theta_trainable=True,
        p_distribution='gauss'
    )

    # Callbacks
    vis = pt.models.VisGLVQ2D(train_ds, show_last_only=False, block=False)

    # Setup trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[
            vis,
            ThetaCallback(train_ds),
            DynamicCallback(),
        ],
        detect_anomaly=True,
    )

    # Training loop
    trainer.fit(model, train_loader)
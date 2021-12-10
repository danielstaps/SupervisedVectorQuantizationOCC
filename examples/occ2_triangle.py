"""GLVQ example using the spiral dataset."""
import argparse

import prototorch as pt
import pytorch_lightning as pl
import torch
from proto.datasets.triangle import Triangle
from proto.functions.losses_csi import occ_csi_soft_loss2
from proto.oneclass import OneClassGMLVQ

if __name__ == "__main__":
    # Command-line arguments
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # Dataset
    num_classes = 2

    train_ds = Triangle(num_samples=1000, dimensions=2, num_classes=1)

    # Dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        num_workers=0,
        batch_size=train_ds.data.shape[0],
    )

    prototypes_per_class = 1
    hparams = dict(
        distribution=(num_classes, prototypes_per_class),
        input_dim=2,
        latent_dim=2,
        proto_lr=0.01,
        bb_lr=0.01,
    )

    # Initialize the model
    model = OneClassGMLVQ(
        hparams,
        optimizer=torch.optim.Adam,
        prototypes_initializer=pt.core.SMCI(train_ds),
        theta_initializer=train_ds.data[train_ds.target == 0],
        loss=occ_csi_soft_loss2,
        theta_trainable=True,
    )

    # Callbacks
    vis = pt.models.VisGLVQ2D(train_ds, show_last_only=False, block=False)

    # Setup trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[
            vis,
        ],
        detect_anomaly=True,
    )

    # Training loop
    trainer.fit(model, train_loader)

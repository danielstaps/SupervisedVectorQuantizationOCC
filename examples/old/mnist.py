"""GLVQ example using the spiral dataset."""

import argparse

import prototorch as pt
import pytorch_lightning as pl
import torch
from prototorch_oneclass.functions.losses import csi_soft_loss
from prototorch_oneclass.oneclass import OneClassGMLVQ
from sklearn.metrics import confusion_matrix
from torchvision import transforms
from torchvision.datasets import MNIST

if __name__ == "__main__":
    # Command-line arguments
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # Dataset
    num_classes = 1

    # Dataset
    train_ds = MNIST(
        "~/datasets",
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ]),
    )
    test_ds = MNIST(
        "~/datasets",
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ]),
    )

    train_ds.data = train_ds.data.flatten(start_dim=1).float()
    test_ds.data = test_ds.data.flatten(start_dim=1).float()

    # Dataloaders
    train_loader = torch.utils.data.DataLoader(train_ds,
                                               num_workers=0,
                                               batch_size=256)
    test_loader = torch.utils.data.DataLoader(test_ds,
                                              num_workers=0,
                                              batch_size=256)

    # Hyperparameters
    prototypes_per_class = 3
    hparams = dict(
        distribution=(num_classes, prototypes_per_class),
        input_dim=train_ds.data.shape[-1],
        latent_dim=2,
        proto_lr=0.01,
        bb_lr=0.01,
    )

    # Initialize the model
    model = OneClassGMLVQ(
        hparams,
        optimizer=torch.optim.Adam,
        prototypes_initializer=pt.core.SSCI(train_loader),
        theta_initializer=train_ds.data[train_ds.targets == 0],
        loss=csi_soft_loss,
        theta_trainable=True,
    )

    # Callbacks
    vis = pt.models.VisGMLVQ2D(train_ds, show_last_only=False, block=False)

    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[
            vis,
        ],
        detect_anomaly=True,
    )
    # Training loop
    trainer.fit(model, train_loader)

    # Testing
    trainer.test(model, test_dataloaders=test_loader)

    # Confusion matrix
    x_test, y_test = test_ds.data, test_ds.targets
    d = model.compute_distances(test_loader)
    y_pred = model.predict_from_distances(d)

    print(confusion_matrix(y_test, y_pred.cpu().numpy()))

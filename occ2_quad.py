"""GLVQ example using the spiral dataset."""
import sys
import argparse

import pytorch_lightning as pl
import torch
import numpy as np

import prototorch as pt
from prototorch.datasets import NumpyDataset

#from proto.datasets.flag import Flag
from proto.oneclass import OneClassGLVQv2, OneClassGMLVQv2, OneClassLGMLVQv2

from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

from proto.datasets.quad import Quad

CUDA = True



if __name__ == "__main__":
    # Command-line arguments
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # Dataset
    num_classes = 1

    #for feature in ["bow_vec", "nat_vec","mif_vec","rmif_vec"]:
    for feature in ["nat_vec"]:

        q = Quad(feature=feature)
        print(q.data, q.target)
        
        print("isnan:",sum(np.isnan(q.data)), sum(np.isnan(q.target)))

        k_split = 10
        d = np.array_split(q.data, k_split)
        t = np.array_split(q.target, k_split)

        if [True for t in t if t.mean() == 1]:
            break

        for k in range(k_split):
            x_test = d[k]
            y_test = t[k]
            x_train = np.concatenate([d[i] for i in range(k_split) if i != k])
            y_train = np.concatenate([t[i] for i in range(k_split) if i != k])
            print(x_train.shape, y_train.shape)
            print(x_test.shape, y_test.shape)


            train_ds = NumpyDataset(x_train, y_train)
            # Dataloaders
            train_loader = torch.utils.data.DataLoader(train_ds,
                                                       num_workers=4,
                                                       batch_size=train_ds.data.shape[0],
                                                       )
           
            test_ds = NumpyDataset(x_test, y_test)
            # Dataloaders
            test_loader = torch.utils.data.DataLoader(test_ds,
                                                      num_workers=4,
                                                      batch_size=test_ds.data.shape[0],
                                                      )           

            # Hyperparameters
            prototypes_per_class = 1
            hparams = dict(
                distribution=(num_classes, prototypes_per_class),
                input_dim=x_train.shape[1],
                latent_dim=2,
                #transfer_function="sigmoid_beta",
                #transfer_beta=10.0,
                proto_lr=0.01,
                bb_lr=0.01,
                #lr=0.01,
            )

            # Initialize the model
            model = OneClassGMLVQv2(
                                    hparams,
                                    optimizer=torch.optim.Adam,
                                    prototypes_initializer=pt.core.SMCI(train_ds),
                                    #prototypes_initializer=pt.core.SSCI(train_ds, noise=5e-2), 
                                    )

            # Callbacks
            vis = pt.models.VisGMLVQ2D(
                train_ds, 
                show_last_only=False, 
                block=False
            )
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
                        vis,
                        # pruning,
                    ],
                    terminate_on_nan=True,
                    gpus='0'
                )
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
           
            # Confusion matrix
            x_train = torch.Tensor(x_train)
            d = model.compute_distances(x_train)
            y_pred = model.predict_from_distances(d)

            print(confusion_matrix(y_train, y_pred.cpu().numpy()))

            # Testing
            trainer.test(model, test_dataloaders=test_loader)
           
            # Confusion matrix
            x_test = torch.Tensor(x_test)
            d = model.compute_distances(x_test)
            y_pred = model.predict_from_distances(d)

            print(confusion_matrix(y_test, y_pred.cpu().numpy()))

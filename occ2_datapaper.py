"""GLVQ example using the spiral dataset."""
import sys
import argparse

import pytorch_lightning as pl
import torch
import numpy as np
import cv2

import prototorch as pt
from prototorch.datasets import NumpyDataset

#from proto.datasets.flag import Flag
from proto.oneclass import OneClassGLVQv2, OneClassGMLVQv2, OneClassLGMLVQv2

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, KFold

import matplotlib.pyplot as plt


CUDA = True



if __name__ == "__main__":
    # Command-line arguments
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # Dataset
    num_classes = 1

    x, y = [], []
    for l, line in enumerate(open('proto/datasets/data_roh/pop_failures.dat','r')):
        if l != 0:
            items = line.rstrip()
            items = [float(i) for i in items.split(" ") if i != '']
            x.append(items[2:-1])
            y.append(items[-1])
    x = np.asarray(x)
    y = np.asarray(y)
    print(x.shape, y.shape)
   
    conf_train, conf_test, acc_train, acc_test = [], [], [], []

    skf = StratifiedKFold(n_splits=10)
    for train, test in skf.split(x, y):
        #print('train -  {}   |   test -  {}'.format(
        #    np.bincount(y[train]), np.bincount(y[test])))

        x_train, y_train = x[train], y[train]
        x_test, y_test = x[test], y[test]

        print(x_train)
        print(y_train)

        train_ds = NumpyDataset(x_train, y_train)
        # Dataloaders
        train_loader = torch.utils.data.DataLoader(train_ds,
                                                   num_workers=4,
                                                   batch_size=train_ds.data.shape[0],
                                                   #batch_size=train_ds.data.shape[0]//10
                                                   #batch_size=1000,
                                                   #batch_size=100
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
            latent_dim=5,
            #transfer_function="sigmoid_beta",
            #transfer_beta=10.0,
            proto_lr=0.0001,
            bb_lr=0.0001,
            #lr=0.01,
        )

        # Initialize the model
        model = OneClassGMLVQv2(
                                hparams,
                                optimizer=torch.optim.Adam,
                                prototypes_initializer=pt.core.SMCI(train_ds),
                                theta_initializer=torch.Tensor(x_train)[y_train == 0],
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
                    #vis,
                    # pruning,
                ],
                terminate_on_nan=True,
                gpus='0'
            )
        else:
            trainer = pl.Trainer.from_argparse_args(
                args,
                callbacks=[
                    #vis,
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

        conf_train.append(confusion_matrix(y_train, y_pred.cpu().numpy()))

        acc_train.append(sum(y_pred.cpu().numpy() == y_train)/len(y_train))

        # Testing
        trainer.test(model, test_dataloaders=test_loader)
       
        # Confusion matrix
        x_test = torch.Tensor(x_test)
        d = model.compute_distances(x_test)
        y_pred = model.predict_from_distances(d)

        conf_test.append(confusion_matrix(y_test, y_pred.cpu().numpy()))
        acc_test.append(sum(y_pred.cpu().numpy() == y_test)/len(y_test))

    for l in range(len(conf_train)):
        print("conf_train:\n",conf_train[l])
        print("acc_train\n",acc_train[l])
        print("conf_test:\n",conf_test[l])
        print("acc_test\n",acc_test[l])

    print("conf_train:\n",np.mean(np.array(conf_train), axis=0))
    print("acc_train\n",np.mean(np.array(acc_train),axis=0))
    print("conf_test:\n",np.mean(np.array(conf_test),axis=0))
    print("acc_test\n",np.mean(np.array(acc_test),axis=0))

    

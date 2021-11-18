"""GLVQ example using the spiral dataset."""

import argparse

import pytorch_lightning as pl
import torch
import numpy as np
import cv2

import prototorch as pt
from prototorch.datasets import NumpyDataset

#from proto.datasets.flag import Flag
from proto.oneclass import OneClassGMLVQv2

from torchvision.datasets import MNIST

from sklearn.datasets import load_digits
from skimage.transform import resize

from keras.datasets import mnist, fashion_mnist, cifar10, cifar100

import matplotlib.pyplot as plt


CUDA = True



def convert_to8x8(data):
    split = 8
    if data.shape[1] == 28:
        data = resize(data, (data.shape[0], 24, 24), anti_aliasing=True)
    #print(data.shape)
    #plt.imshow(data[0], cmap='gray')
    #plt.show()
    data = np.hsplit(data, split)
    for i, img_row in enumerate(data):
        data[i] = np.dsplit(img_row, split)
        #print([[d.shape for d in data[i]] for i in range(len(data))])
    for i in range(len(data)):
        for j in range(len(data[i])):
            #print(data[i][j].shape)
            data[i][j] = np.sum(data[i][j], axis=(1,2))
            data[i][j] = np.expand_dims(data[i][j], axis=-1)
            data[i][j] = np.expand_dims(data[i][j], axis=-1)
        data[i] = np.concatenate(data[i], axis=-1)
        #data[i] = np.expand_dims(data[i], axis=-1)
    data = np.concatenate(data, axis=-2)
    #plt.imshow(data[0], cmap='gray')
    #plt.show()
    #print(data.shape)
    return data



def give_data_back():
    rgb_weights = [0.2989, 0.5870, 0.1140]
    x, y, xt, yt = [], [], [], []
    for i, fct in enumerate([mnist, fashion_mnist, cifar10, cifar100]):
        (x_train, y_train), (x_test, y_test) = fct.load_data()
        #print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
        if len(x_train.shape) == 4:
            #plt.imshow(x_train[0])
            #plt.show()
            x_train = np.dot(x_train, rgb_weights)
            x_test = np.dot(x_test, rgb_weights)
            #print(x_train.shape, x_test.shape)
        #plt.imshow(x_train[0], cmap='gray')
        #plt.show()
        x_train = convert_to8x8(x_train)
        x_test = convert_to8x8(x_test)
        y_train = y_train.squeeze()
        y_test = y_test.squeeze()

        if i == 0:
            y_train = np.zeros(y_train.shape)
            y_test = np.zeros(y_test.shape)
        else:
            y_train = np.ones(y_train.shape)
            y_test = np.ones(y_test.shape)

        x.append(x_train)
        y.append(y_train)
        xt.append(x_test)
        yt.append(y_test)

    x = np.concatenate(x, axis=0)
    y = np.concatenate(y, axis=0)
    xt = np.concatenate(xt, axis=0)
    yt = np.concatenate(yt, axis=0)
    
    #plt.imshow(x[0], cmap='gray')
    #plt.show()

    x = np.resize(x, (x.shape[0], 8*8))
    xt = np.resize(xt, (xt.shape[0], 8*8))

    #plt.imshow(np.expand_dims(x[0], axis=0), cmap='gray')
    #plt.show()


    print(x.shape, y.shape, xt.shape, yt.shape)
    return (x, y), (xt, yt)




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
    train_loader = torch.utils.data.DataLoader(train_ds,
                                               num_workers=4,
                                               #batch_size=train_ds.data.shape[0],
                                               batch_size=100
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
    )

    # Initialize the model
    model = OneClassGMLVQv2(hparams,
                            optimizer=torch.optim.Adam,
                            #prototypes_initializer=pt.core.SMCI(train_ds),
                            prototypes_initializer=pt.core.SSCI(train_ds, noise=5e-2), 
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

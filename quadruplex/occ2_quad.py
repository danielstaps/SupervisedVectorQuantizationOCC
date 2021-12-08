import os
import sys
import pickle
import argparse

import pytorch_lightning as pl
import torch
import numpy as np
import math
from itertools import product
from sortedcontainers import SortedList
from datetime import datetime

import prototorch as pt
from prototorch.datasets import NumpyDataset

#from proto.datasets.flag import Flag
from proto.oneclass import OneClassGLVQv2, OneClassGMLVQv2, OneClassLGMLVQv2

from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

from proto.datasets.quad import Quad

from torch.optim.lr_scheduler import ExponentialLR


CUDA = False


if __name__ == "__main__":
    # Command-line arguments
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    #now = datetime.now()
    #current_time = now.strftime("%Y-%m-%d-%H-%M-%S")
    current_time = "2021-12-08_brier-original"
    if not os.path.isdir(current_time):
        os.mkdir(current_time)

    # Dataset
    num_classes = 1
    for ta in [[12,1],[9,1],[6,1],[3,1]]:
        #for ta in [[4,1],[6,1],[2,2],[4,2],[6,2],[2,3],[4,3],[6,3]]:

        latent_dim = ta[0]
        prototypes_per_class = ta[1]

        for feature in ["bow_vec", "nat_vec","mif_vec","rmif_vec"]:
            #for feature in ["mif_vec", "rmif_vec"]:
            
            st_filename = 'CCM_' + feature + '_dim' + str(latent_dim) + '_p' + str(prototypes_per_class) + '_e' + str(args.max_epochs)
            
            q = Quad(feature=feature)
            #print(q.data, q.target)

            #print("isnan:",sum(np.isnan(q.data)), sum(np.isnan(q.target)))

            k_split = 10
            d = np.array_split(q.data, k_split)
            t = np.array_split(q.target, k_split)

            if [True for t in t if t.mean() == 1]:
                break

            omega_matrizes = []
            conf_mat_acc = []
            conf_mat_acc.append(['train_acc','test_acc','train_conf','test_conf'])

            for k in range(k_split):
                print("\n\n")
                print(f"k:{k}, feature:{feature}, protos:{prototypes_per_class}, latent:{latent_dim}")

                x_test = d[k]
                y_test = t[k]
                x_train = np.concatenate([d[i] for i in range(k_split) if i != k])
                y_train = np.concatenate([t[i] for i in range(k_split) if i != k])
                #print(x_train.shape, y_train.shape)
                #print(x_test.shape, y_test.shape)
                x_train_addition = np.repeat(x_train[y_train == 0,:], 2, axis=0)
                y_train_addition = np.repeat(y_train[y_train == 0], 2)
                x_train = np.append(x_train, x_train_addition, axis=0)
                y_train = np.append(y_train, y_train_addition)
                print(x_train.shape, y_train.shape)

                print(f"class distribution in train and test:[{len(y_train)-sum(y_train),sum(y_train)}], [{len(y_test)-sum(y_test),sum(y_test)}]")


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
                hparams = dict(
                        distribution=(num_classes, prototypes_per_class),
                        input_dim=x_train.shape[1],
                        latent_dim=latent_dim,
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
                        theta_initializer=x_train,
                        #prototypes_initializer=pt.core.SSCI(train_ds, noise=5e-2), 
                        lr_scheduler=ExponentialLR,
                        lr_scheduler_kwargs=dict(gamma=0.99, verbose=False),
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
                distances = model.compute_distances(x_train)
                y_pred = model.predict_from_distances(distances)

                train_acc = trainer.validate(model, train_loader)[0]['val_acc']
                train_confmat = confusion_matrix(y_train, y_pred.cpu().numpy())
                print(train_confmat)

                # Testing
                trainer.test(model, test_dataloaders=test_loader)

                # Confusion matrix
                x_test = torch.Tensor(x_test)
                distances = model.compute_distances(x_test)
                y_pred = model.predict_from_distances(distances)

                test_acc = trainer.validate(model, test_loader)[0]['val_acc']
                test_confmat = confusion_matrix(y_test, y_pred.cpu().numpy())
                print(test_confmat)

                conf_mat_acc.append([train_acc,test_acc,train_confmat,test_confmat])

                omega_matrizes.append(model.omega_matrix)

                def plot_matrix(matrix):
                    title = "Lambda matrix"
                    fig = plt.figure(title)
                    #ax = plt.gca()
                    im = plt.imshow(matrix, cmap='RdBu_r', vmin=-0.3, vmax=0.3)
                    #fig.show()
                    #plt.pause(1)
                    # plt.show(block=False)
                    name = current_time + "/runs/" + str(k) + "_" + st_filename + ".png"

                    plt.savefig(name)
                    plt.figure().clear()
                    plt.close()
                    plt.cla()
                    plt.clf()

                omegaMat = model.omega_matrix
                lamMat = omegaMat @ omegaMat.T

                if not os.path.isdir(current_time + "/runs"):
                    os.mkdir(current_time + "/runs")
                np.savez(current_time + "/runs/" + str(k) + "_" + st_filename, lamMat)
                plot_matrix(lamMat)
    
            if not os.path.isdir(current_time + "/summary"):
                os.mkdir(current_time + "/summary")
            with open(current_time + "/summary/" + st_filename+'.pkl', 'wb') as pkl:
                pickle.dump(conf_mat_acc, pkl)
            with open(current_time + "/summary/" + st_filename + ".txt", 'w') as txt:
                for line in conf_mat_acc:
                    for element in line:
                        print(element)
                        txt.write(str(element) + "\n")
                    txt.write("\n")
            
            omega_stacked = np.stack(omega_matrizes)
            omega_mean = np.mean(omega_stacked, axis=0)

            lambdaMatrix = omega_mean @ omega_mean.T

            if not os.path.isdir(current_time + "/meanCCM"):
                os.mkdir(current_time + "/meanCCM")
            np.savez(current_time + "/meanCCM/" + 'mean' + st_filename,lambdaMatrix)

            title = "Lambda matrix"
            fig = plt.figure(title)
            ax = plt.gca()
            im = plt.imshow(lambdaMatrix, cmap='RdBu_r', vmin=-0.3, vmax=0.3)

            cbar = ax.figure.colorbar(im, ax=ax)
            #cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
            nuc = ['A', 'C', 'G', 'T']
            if feature == 'rmif_vec':
                col_labels = SortedList()
                tau = int(np.shape(lambdaMatrix)[1] / 4)

                for letter in nuc:
                    for ii in range(tau):
                        textbaustein = '(' + letter + ',' + str(ii + 1) + ')'
                        col_labels.add(textbaustein)
                xBezeichnung = " "
            elif feature == 'mif_vec':
                col_labels = SortedList()
                tau = np.shape(lambdaMatrix)[1]
                xBezeichnung = "$\tau$"
                for ii in range(tau):
                    textbaustein = '(X,' + str(ii + 1) + ')'
                    col_labels.add(textbaustein)
            elif feature == 'nat_vec':
                col_labels = []
                numMoments = int(np.shape(lambdaMatrix)[1] / 4)
                xBezeichnung = " "
                for letter in nuc:
                    for ii in range(numMoments):
                        textbaustein = "$m^" + str(ii) + "_" + letter + "$"
                        print(textbaustein)
                        col_labels.append(textbaustein)
            elif feature == 'bow_vec':
                col_labels = []
                k = int(math.log(np.shape(lambdaMatrix)[1], 4))
                xBezeichnung = " "
                perms = [''.join(w) for w in list(product(nuc, repeat=k))]
                for ii in perms:
                    col_labels.append(ii)
            else:
                col_labels = []

            ax.set_xticks(np.arange(len(col_labels)))
            ax.set_yticks(np.arange(len(col_labels)))
            # ... and label them with the respective list entries.
            ax.set_xticklabels(col_labels, fontsize=15)
            ax.set_yticklabels(col_labels, fontsize=15)
            ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

            # Rotate the tick labels and set their alignment.
            if feature != 'mif_vec':
                plt.setp(ax.get_xticklabels(),
                         rotation=45,
                         ha="left",
                         rotation_mode="anchor")

            ax.spines[:].set_visible(False)

            ax.set_xticks(np.arange(len(col_labels) + 1) - .5, minor=True)
            ax.set_yticks(np.arange(len(col_labels) + 1) - .5, minor=True)
            ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
            ax.tick_params(which="minor", bottom=False, left=False)

            plt.xlabel(xBezeichnung)
            '''''
            mng = plt.get_current_fig_manager()
            mng.full_screen_toggle()
            '''
            #fig.show()
            #plt.pause(1)
            name = current_time + "/meanCCM/" + 'mean'+ st_filename + '.png'
            plt.savefig(name)
            plt.figure().clear()
            plt.close()
            plt.cla()
            plt.clf()


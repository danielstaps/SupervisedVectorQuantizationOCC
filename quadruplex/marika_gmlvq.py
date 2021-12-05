import argparse
import numpy as np
import json
from datetime import datetime
from prototorch.datasets.abstract import NumpyDataset
import torch
import prototorch as pt
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedShuffleSplit
from sortedcontainers import SortedList
from torch.nn.parameter import Parameter

#achtung  neu
from itertools import product
import math


def run_kfold_training(dataset,
                       model_class,
                       model_kwargs,
                       trainer_kwargs,
                       k=5,
                       validation_function=None,
                       addionalInformation=' ',
                       shuffle=True):

    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    kfold = StratifiedShuffleSplit(n_splits=k)

    fold_log = []
    omega_all = []

    for fold, (train_ids, test_ids) in enumerate(
            kfold.split(dataset.data, dataset.targets)):
        # Fold
        print(f"FOLD {fold+1}")
        print("--------------------------------")

        blubb = np.repeat(train_ids[dataset.targets[train_ids] == 0], 3)
        train_ids = np.append(train_ids, blubb)

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        trainloader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=11,
                                                  sampler=train_subsampler)
        testloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=11,
                                                 sampler=test_subsampler)

        if issubclass(model_kwargs.get('prototypes_initializer_class'),
                      pt.initializers.ShapeAwareCompInitializer):
            model_kwargs["prototypes_initializer"] = model_kwargs[
                "prototypes_initializer_class"](model_kwargs.get("data_dim"))
        if issubclass(model_kwargs.get('prototypes_initializer_class'),
                      pt.initializers.AbstractClassAwareCompInitializer):
            model_kwargs["prototypes_initializer"] = model_kwargs[
                "prototypes_initializer_class"](trainloader)
        else:
            print('Something did not work ;(')

        model = model_class(**model_kwargs)

        # for better omega initialization
        cov_mat = np.cov(dataset.data.T)
        eig_val, eig_vec = np.linalg.eig(cov_mat)

        om_mat = torch.tensor(
            (eig_vec.T[:][:model_kwargs['hparams']['latent_dim']]).T,
            dtype=torch.float32)
        model.register_parameter("_omega", Parameter(om_mat))

        trainer = Trainer.from_argparse_args(args, **trainer_kwargs)
        trainer.fit(model, trainloader)

        fold_log.append(trainer.validate(model, testloader))
        omega_all.append(model.omega_matrix)

        if validation_function is not None:
            validation_function(model, trainloader, testloader,
                                addionalInformation)
        #print(fold_log)

    return fold_log, omega_all


def plot_matrix(matrix, addionalInformation):
    title = "Lambda matrix"
    fig = plt.figure(title)
    ax = plt.gca()
    im = plt.imshow(matrix, cmap='RdBu_r', vmin=-0.3, vmax=0.3)

    fig.show()
    plt.pause(1)

    # plt.show(block=False)

    now = datetime.now()
    current_time = now.strftime("%m-%d-%Y-%H-%M-%S")
    name = 'CCM_' + addionalInformation['which_features'] + '_dim' + str(
        addionalInformation['latent_dim']) + '_p' + str(
            addionalInformation['prototypes_per_class']) + '_e' + str(
                addionalInformation['max_epoch']) + current_time + ".png"

    plt.savefig(name)


def show_validation(model, trainloader, validationloader, addionalInformation):

    for name, loader in [("train", trainloader),
                         ("validation", validationloader)]:
        target = []
        pred = []

        for x, y in loader:
            target.extend(y.tolist())
            pred.extend(model.predict(x).tolist())

        print(name)
        print(confusion_matrix(target, pred))  #, normalize="all"
    # Training loop
    omegaMat = model.omega_matrix  # lernen omega mat, visualisiren lambda mat
    lamMat = omegaMat @ omegaMat.T

    if addionalInformation != ' ':
        np.savez(
            'CCM_' + addionalInformation['which_features'] + '_dim' +
            str(addionalInformation['latent_dim']) + '_p' +
            str(addionalInformation['prototypes_per_class']) + '_e' +
            str(addionalInformation['max_epoch']), lamMat)
    else:
        np.savez('Omega_mif4', lamMat)

    plot_matrix(lamMat, addionalInformation)


class VisGMLVQ2D(pt.models.vis.Vis2DAbstract):
    def __init__(self, *args, ev_proj=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.ev_proj = ev_proj

    def on_epoch_end(self, trainer, pl_module):
        if not self.precheck(trainer):
            return True

        protos = pl_module.prototypes
        plabels = pl_module.prototype_labels
        x_train, y_train = self.x_train, self.y_train
        device = pl_module.device
        omega = pl_module._omega.detach()
        lam = omega @ omega.T
        u, _, _ = torch.pca_lowrank(lam, q=2)
        with torch.no_grad():
            x_train = torch.Tensor(x_train).to(device)
            x_train = x_train @ u
            x_train = x_train.cpu().detach()
        if self.show_protos:
            with torch.no_grad():
                protos = torch.Tensor(protos).to(device)
                protos = protos @ u
                protos = protos.cpu().detach()
        ax = self.setup_ax()
        self.plot_data(ax, x_train, y_train)
        if self.show_protos:
            self.plot_protos(ax, protos, plabels)

        self.log_and_display(trainer, pl_module)

    def plot_data(self, ax, x, y):
        ax.scatter(
            x[:, 0],
            x[:, 1],
            c=y,
            alpha=0.4,
            cmap=self.cmap,
            #edgecolor="k",
            marker="o",
            s=30,
        )


if __name__ == "__main__":

    #hier habe ich angefangen zu ändern und erstmal alle Paramter die man so ändern könnte hingeschrieben, das macht es einfacher den übersicht zu händeln
    # Hyperparameters
    data_name = "data_ground_truth_bow_vec_three_classes.npz"
    which_features = "bow_vec"
    prototypes_per_class = 2
    max_epoch = 10
    latent_dim = 2

    addionalInformation = {}
    addionalInformation['which_features'] = which_features
    addionalInformation['data_name'] = data_name
    addionalInformation['prototypes_per_class'] = prototypes_per_class
    addionalInformation['max_epoch'] = max_epoch
    addionalInformation['latent_dim'] = latent_dim

    # Dataset
    container = np.load(data_name)
    x_train = container[which_features]
    x_train = np.array([json.loads(entry) for entry in x_train])
    #x_train = x_train/np.max(x_train, axis = 0)
    y_train = container["converted_labels"]
    train_ds = NumpyDataset(x_train, y_train)

    print('Number of input dimensions', np.shape(x_train)[1])
    input_dim = np.shape(x_train)[1]

    num_classes = np.unique(y_train).shape[0]
    addionalInformation['num_classes'] = num_classes

    hparams = dict(
        input_dim=input_dim,  # Vektorlänge
        latent_dim=latent_dim,  # vis dim
        distribution=(num_classes, prototypes_per_class),
        proto_lr=0.01,
        bb_lr=0.001,  # bb = matrix, lr = learning rate
    )

    prototypes_initializer = pt.initializers.SMCI
    model_kwargs = {
        "hparams": hparams,
        "optimizer": torch.optim.Adam,
        "prototypes_initializer_class": prototypes_initializer,
    }

    vis = VisGMLVQ2D(data=train_ds)

    # Trainer Definition
    trainer_kwargs = dict(
        callbacks=[vis],
        max_epochs=max_epoch)  #callbacks=[vis] if vis is activated

    # Run Training
    logs, omega_all = run_kfold_training(
        train_ds,
        pt.models.GMLVQ,
        model_kwargs,
        trainer_kwargs,
        k=10,  #fold number can be changed
        validation_function=show_validation,
        addionalInformation=addionalInformation)

    print(logs)

    omega_stacked = np.stack(omega_all)
    mean_omega = np.mean(omega_stacked, axis=0)

    lamMat = mean_omega @ mean_omega.T
    np.savez(
        'meanCCM_' + addionalInformation['which_features'] + '_dim' +
        str(addionalInformation['latent_dim']) + '_p' +
        str(addionalInformation['prototypes_per_class']) + '_e' +
        str(addionalInformation['max_epoch']), lamMat)

    title = "Lambda matrix"
    fig = plt.figure(title)
    ax = plt.gca()
    im = plt.imshow(lamMat, cmap='RdBu_r', vmin=-0.3, vmax=0.3)

    cbar = ax.figure.colorbar(im, ax=ax)
    #cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    nuc = ['A', 'C', 'G', 'T']
    if addionalInformation['which_features'] == 'rmif_vec':
        col_labels = SortedList()
        tau = int(np.shape(lamMat)[1] / 4)

        for letter in nuc:
            for ii in range(tau):
                textbaustein = '(' + letter + ',' + str(ii + 1) + ')'
                col_labels.add(textbaustein)
        xBezeichnung = " "
    elif addionalInformation['which_features'] == 'mif_vec':
        col_labels = SortedList()
        tau = np.shape(lamMat)[1]
        xBezeichnung = "$\tau$"
        for ii in range(tau):
            textbaustein = '(X,' + str(ii + 1) + ')'
            col_labels.add(textbaustein)
    elif addionalInformation['which_features'] == 'nat_vec':
        col_labels = []
        numMoments = int(np.shape(lamMat)[1] / 4)
        xBezeichnung = " "
        for letter in nuc:
            for ii in range(numMoments):
                textbaustein = "$m^" + str(ii) + "_" + letter + "$"
                print(textbaustein)
                col_labels.append(textbaustein)
    elif addionalInformation['which_features'] == 'bow_vec':
        col_labels = []
        k = int(math.log(np.shape(lamMat)[1], 4))
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
    if addionalInformation['which_features'] != 'mif_vec':
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
    fig.show()
    plt.pause(1)

    name = 'meanCCM_ ' + addionalInformation['which_features'] + '_dim' + str(
        addionalInformation['latent_dim']) + '_p' + str(
            addionalInformation['prototypes_per_class']) + '_e' + str(
                addionalInformation['max_epoch']) + '.png'
    plt.savefig(name)

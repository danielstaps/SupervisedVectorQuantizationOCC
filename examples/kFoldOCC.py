import os
import pickle
from datetime import datetime

import numpy as np
import prototorch as pt
import pytorch_lightning as pl
import torch
from proto.functions.callbacks import ThetaCallback
from proto.oneclass import OneClassGLVQ, OneClassGMLVQ, OneClassLGMLVQ
from prototorch.datasets import NumpyDataset
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, LeaveOneOut, StratifiedKFold


def train_fct(x, y, train, test, results, params, args, model_type):

    x_train, y_train = x[train], y[train]
    x_test, y_test = x[test], y[test]

    print(x_train)
    print(y_train)

    train_ds = NumpyDataset(x_train, y_train)
    # Dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        num_workers=4,
        batch_size=train_ds.data.shape[0],
        #batch_size=train_ds.data.shape[0]//10
        #batch_size=1000,
        #batch_size=100
    )

    test_ds = NumpyDataset(x_test, y_test)
    # Dataloaders
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        num_workers=4,
        batch_size=test_ds.data.shape[0],
    )

    # define model
    model_fcts = {
        'normal': OneClassGLVQ,
        'mapping': OneClassGMLVQ,
        'localmapping': OneClassLGMLVQ
    }
    OCCmodel = model_fcts[model_type]

    hparams = dict(
        distribution=(params['num_classes'], params['prototypes_per_class']),
        input_dim=x_train.shape[1],
        latent_dim=params['latent_dim'],
        #transfer_function="sigmoid_beta",
        #transfer_beta=10.0,
        proto_lr=0.0001,
        bb_lr=0.0001,
        #lr=0.01,
    )

    # Initialize the model
    model = OCCmodel(
        hparams,
        optimizer=torch.optim.Adam,
        prototypes_initializer=pt.core.SMCI(train_ds),
        theta_initializer=torch.Tensor(x_train)[y_train == 0],
        #prototypes_initializer=pt.core.SSCI(train_ds, noise=5e-2),
        omega_initializer=pt.core.PCALTI(torch.Tensor(x_train)),
    )

    # Callbacks
    vis = pt.models.VisGMLVQ2D(train_ds, show_last_only=False, block=False)
    pruning = pt.models.PruneLoserPrototypes(
        threshold=0.01,
        idle_epochs=1,
        prune_quota_per_epoch=1,
        frequency=1,
        verbose=True,
    )
    theta = ThetaCallback()

    # Setup trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[
            theta,
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

    if 'conf_train' not in results.keys():
        results['conf_train'] = []
    results['conf_train'].append(
        confusion_matrix(y_train,
                         y_pred.cpu().numpy()))
    if 'acc_train' not in results.keys():
        results['acc_train'] = []
    results['acc_train'].append(
        sum(y_pred.cpu().numpy() == y_train) / len(y_train))

    # Testing
    trainer.test(model, test_dataloaders=test_loader)

    # Confusion matrix
    x_test = torch.Tensor(x_test)
    d = model.compute_distances(x_test)
    y_pred = model.predict_from_distances(d)

    if 'conf_test' not in results.keys():
        results['conf_test'] = []
    results['conf_test'].append(confusion_matrix(y_test, y_pred.cpu().numpy()))
    if 'acc_test' not in results.keys():
        results['acc_test'] = []
    results['acc_test'].append(
        sum(y_pred.cpu().numpy() == y_test) / len(y_test))

    if 'omega_matrix' not in results.keys():
        results['omega_matrix'] = []
    results['omega_matrix'].append(model.omega_matrix)

    return results


def stratified_kfold(fct, x, y, params, args, model_type):
    results = {}
    skf = StratifiedKFold(n_splits=10)
    for train, test in skf.split(x, y):
        print('train -  {}   |   test -  {}'.format(np.bincount(y[train]),
                                                    np.bincount(y[test])))
        results = train_fct(x, y, train, test, results, params, args,
                            model_type)
    return results


def leaveoneout(fct, x, y, params, args, model_type):
    results = {}
    loo = LeaveOneOut()
    for train, test in loo.split(X):
        print("%s %s" % (train, test))
        results = train_fct(x, y, train, test, results, params, args,
                            model_type)
    return results


def kFoldOcc(data, params, args, model_type='mapping', experiment_name=''):
    # extract data
    if type(data) == tuple:
        (x, y) = data
        y = np.asarray(y, dtype=int)

    # kfold fcts
    print("Class distributions:", np.bincount(y))
    min_c = np.amin(np.bincount(y))
    if min_c <= 30:
        print(
            f"One Class has only {min_c} datapoints, switch to LEAVE ONE OUT")
        results = leaveoneout(train_fct, x, y, params, args, model_type)
    else:
        print(
            f"Classes have enough data, switch to STRATIFIED K FOLD with K=10")
        results = stratified_kfold(train_fct, x, y, params, args, model_type)

    for l in range(len(results['conf_train'])):
        print("conf_train:\n", results['conf_train'][l])
        print("acc_train\n", results['acc_train'][l])
        print("conf_test:\n", results['conf_test'][l])
        print("acc_test\n", results['acc_test'][l])
        #print("omega\n",results['omega_matrix'])

    print("conf_train:\n", np.mean(np.array(results['conf_train']), axis=0))
    print("acc_train\n", np.mean(np.array(results['acc_train']), axis=0))
    print("conf_test:\n", np.mean(np.array(results['conf_test']), axis=0))
    print("acc_test\n", np.mean(np.array(results['acc_test']), axis=0))

    name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S" + experiment_name)
    if not os.path.isdir('results'):
        os.mkdir('results')
    with open('results/' + name + '.pkl', 'wb') as pklfile:
        pickle.dump(results, pklfile)

import argparse
import numpy as np
from kFoldOCC import kFoldOcc

import pytorch_lightning as pl


if __name__ == "__main__":
    # Command-line arguments
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()


    # Dataset
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
    data = (x, y)
   

    # Hyperparameters
    num_classes = 1
    prototypes_per_class = 1
    latent_dim = 5
    hparams = {
            'num_classes':num_classes,
            'prototypes_per_class':prototypes_per_class,
            'latent_dim':latent_dim,
            }

    # Train the model
    kFoldOcc(data, hparams, args, model_type='mapping')

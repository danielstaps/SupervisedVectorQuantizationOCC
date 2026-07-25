"""Smoke test: the SVQ-OCC one-class classifier trains on the bundled Flag set."""

import prototorch as pt
import pytorch_lightning as pl
import torch

from prototorch_oneclass import SVQ_OCC
from prototorch_oneclass.datasets import Flag


def test_svqocc_trains():
    ds = Flag()
    loader = torch.utils.data.DataLoader(ds, batch_size=len(ds), shuffle=True)
    model = SVQ_OCC(
        dict(distribution=(1, 4), lr=0.01),
        prototypes_initializer=pt.core.SSCI(ds),
        theta_initializer=ds,
    )
    trainer = pl.Trainer(
        accelerator="cpu",
        max_epochs=20,
        enable_progress_bar=False,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
    )
    trainer.fit(model, loader)  # trains without error
    assert model.proto_layer._components.shape[0] == 4

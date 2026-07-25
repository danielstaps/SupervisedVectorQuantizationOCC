""" One Class Classifier based on GLVQ framework """

from functools import partial

import torch
import torchmetrics
from prototorch.core.distances import squared_euclidean_distance
from prototorch.models.glvq import GLVQ
from prototorch.nn import LambdaLayer
from torch.nn.parameter import Parameter

from .functions.competitions import WTAC_Thresh
from .functions.losses import brier_score


def get_theta(train_ds, model):
    x_train, y_train = train_ds.data, train_ds.target
    d = model.compute_distances(x_train)

    _, plabels = model.proto_layer()

    theta = torch.zeros(len(plabels))
    quantile = 0.33

    for i, label in enumerate(plabels):
        idx = torch.argmin(d[y_train == label], dim=1) == i
        t_value = torch.quantile(d[y_train == label, i], quantile)
        if sum(idx) == 0:
            theta[i] = torch.sqrt(t_value)
        else:
            theta[i] = torch.quantile(d[y_train == label, i][idx], quantile)
            if theta[i] <= torch.sqrt(t_value):
                theta[i] = torch.sqrt(t_value)
    return theta


class SVQ_Initialization:
    def __init__(self, hparams, **kwargs):
        # Collect Arguments
        loss = kwargs.pop("loss", brier_score)
        self.p_distribution = kwargs.pop("p_distribution", None)
        self.score = kwargs.pop("score", None)
        self.theta_trainable = kwargs.pop("theta_trainable", True)

        train_ds = kwargs.pop("theta_initializer")
        if train_ds is None:
            raise NotImplementedError("No default theta initializer")

        # Initialize Theta
        theta = get_theta(train_ds, self)

        self.register_parameter(
            "_theta",
            Parameter(
                theta,
                requires_grad=self.theta_trainable,
            ),
        )

        self.register_parameter(
            "_sigma",
            Parameter(
                torch.Tensor([1.0]),
                requires_grad=False,
            ),
        )

        self.register_parameter(
            "_ng_lambda",
            Parameter(
                torch.Tensor([0.5]),
                requires_grad=False,
            ),
        )

        self.register_parameter(
            "_alpha",
            Parameter(
                torch.Tensor([1.0]),
                requires_grad=False,
            ),
        )

        self.register_parameter(
            "_zero_mean",
            Parameter(
                torch.Tensor([0.0]),
                requires_grad=False,
            ),
        )

        # Layers
        self.loss = LambdaLayer(
            partial(
                loss,
                theta_boundary=self._theta,
                distribution=self.p_distribution,
                score=self.score,
                sigma=self._sigma,
                ng_lambda=self._ng_lambda,
                alpha=self._alpha,
                zero_mean=self._zero_mean,
            ),
            name=loss.__name__,
        )
        self.competition_layer = WTAC_Thresh(theta_boundary=self._theta)

    @property
    def theta_boundary(self):
        return self._theta.detach().cpu()

    @property
    def scale(self):
        return self._scale.detach().cpu()


class SVQ_OCC(
    GLVQ,
    SVQ_Initialization,
):
    def __init__(self, hparams, **kwargs) -> None:
        GLVQ.__init__(self, hparams, **kwargs)
        SVQ_Initialization.__init__(self, hparams, **kwargs)

        distance_fn = kwargs.get("distance_fn", squared_euclidean_distance)
        self.distance_layer = LambdaLayer(distance_fn)

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def log_acc(self, distances, targets, tag):
        preds = self.predict_from_distances(distances)
        accuracy = torchmetrics.functional.accuracy(
            preds.int(),
            targets.int(),
            "multiclass",
            num_classes=self.num_classes + 1,
        )

        self.log(
            tag,
            accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def test_step(self, batch, batch_idx):
        x, targets = batch

        preds = self.predict(x)
        accuracy = torchmetrics.functional.accuracy(
            preds.int(),
            targets.int(),
            "multiclass",
            num_classes=self.num_classes + 1,
        )

        self.log("test_acc", accuracy)

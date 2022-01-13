""" One Class Classifier based on GLVQ framework """

from functools import partial

import torch
from prototorch.models.glvq import GLVQ, GMLVQ, LGMLVQ
from prototorch.nn import LambdaLayer
from torch.nn.parameter import Parameter

from .functions.competitions import WTAC_Thresh
from .functions.losses import csi_soft_loss


def get_theta(train_ds, model):
    x_train, y_train = train_ds.data, train_ds.target
    d = model.compute_distances(x_train)

    _, plabels = model.proto_layer()

    theta = torch.zeros(len(plabels))
    quantile = 0.33

    for i, label in enumerate(plabels):
        idx = torch.argmin(d[y_train == label], dim=1) == i
        theta[i] = torch.quantile(d[y_train == label, i][idx], quantile)

    return theta


class OneClassInitialization:
    def __init__(self, backbone, hparams, **kwargs):
        # Collect Arguments
        loss = kwargs.pop("loss", csi_soft_loss)
        self.p_distribution = kwargs.pop("p_distribution", None)
        self.score = kwargs.pop("score", None)
        theta_trainable = kwargs.pop("theta_trainable", True)

        train_ds = kwargs.pop("theta_initializer")
        if train_ds is None:
            raise NotImplementedError("No default theta initializer")

        # Initialize Theta
        theta = get_theta(train_ds, self)
        gamma = get_theta(train_ds, self)

        self.register_parameter(
            "_theta",
            Parameter(
                theta,
                requires_grad=theta_trainable,
            ),
        )

        self.register_parameter(
            "_gamma",
            Parameter(
                gamma,
                requires_grad=True,
            ),
        )

        self.register_parameter(
            "_sigma",
            Parameter(
                torch.Tensor([1.]),
                requires_grad=False,
            ),
        )

        # Layers
        self.loss = LambdaLayer(
            partial(loss,
                    theta_boundary=self._theta,
                    distribution=self.p_distribution,
                    score=self.score,
                    gamma=self._gamma,
                    sigma=self._sigma,
                    backbone=backbone),
            name=loss.__name__,
        )
        self.competition_layer = WTAC_Thresh(theta_boundary=self._theta)

    def configure_optimizers(self):
        proto_opt = self.optimizer(self.proto_layer.parameters(),
                                   lr=self.hparams.proto_lr)
        # Only add a backbone optimizer if backbone has trainable parameters
        bb_params = list(self.backbone.parameters())
        if (bb_params):
            bb_opt = self.optimizer(bb_params, lr=self.hparams.bb_lr)
            optimizers = [proto_opt, bb_opt]
        else:
            optimizers = [proto_opt]
        theta_opt = self.optimizer(self._theta, lr=self.hparams.theta_lr)
        optimizers.append(theta_opt)
        return optimizers

    @property
    def theta_boundary(self):
        return self._theta.detach().cpu()


class OneClassGLVQ(
        GLVQ,
        OneClassInitialization,
):
    def __init__(self, hparams, **kwargs) -> None:
        GLVQ.__init__(self, hparams, **kwargs)
        OneClassInitialization.__init__(self, hparams, **kwargs)


class OneClassGMLVQ(
        GMLVQ,
        OneClassInitialization,
):
    def __init__(self, hparams, **kwargs) -> None:
        GMLVQ.__init__(self, hparams, **kwargs)
        OneClassInitialization.__init__(self, self._omega, hparams, **kwargs)

        self._omega.requires_grad = False


class OneClassLGMLVQ(
        LGMLVQ,
        OneClassInitialization,
):
    def __init__(self, hparams, **kwargs) -> None:
        LGMLVQ.__init__(self, hparams, **kwargs)
        OneClassInitialization.__init__(self, hparams, **kwargs)

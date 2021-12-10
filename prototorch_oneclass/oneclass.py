""" One Class Classifier based on GLVQ framework """

import numpy as np
import torch
from prototorch.core.distances import (lomega_distance, omega_distance,
                                       squared_euclidean_distance)
from prototorch.models.glvq import GLVQ, GMLVQ, LGMLVQ
from prototorch.nn import LambdaLayer
from torch.nn.parameter import Parameter
from torch.optim.lr_scheduler import ExponentialLR

from .functions.competitions import wtac_thresh
from .functions.losses_csi import occ_csi_soft_loss2


class ThetaInitializer():
    def __init__(self, x_train, model):
        if type(x_train) == np.ndarray:
            x_train = torch.Tensor(x_train)
        d = model.compute_distances(x_train).detach().cpu()
        self.theta = torch.abs(torch.mean(d, axis=0))


class OneClassMixin():
    def init_variant(
        self,
        theta_init,
        theta_trainable=True,
        loss=occ_csi_soft_loss2,
    ):
        print("label_shape:", self.proto_layer.labels.shape)
        self.register_parameter(
            "_theta",
            Parameter(
                theta_init.theta,
                requires_grad=theta_trainable,
            ),
        )
        self.loss = LambdaLayer(loss)
        self.wtac = wtac_thresh

    def init_params(self, ):
        self.lr_scheduler = ExponentialLR
        self.lr_scheduler_kwargs = dict(gamma=0.99, verbose=False)

    def shared_step(self, batch, batch_idx, optimizer_idx=None):
        x, y = batch
        out = self.compute_distances(x)
        plabels = self.proto_layer.labels
        loss = self.loss(
            out,
            y,
            prototype_labels=plabels,
            theta_boundary=self._theta,
            device=self.device,
        )
        return out, loss

    @property
    def theta_boundary(self):
        return self._theta.detach().cpu()

    def predict_from_distances(self, distances):
        with torch.no_grad():
            _, plabels = self.proto_layer()
            y_pred = self.wtac(distances, plabels, self._theta)
        return y_pred


class OneClassGLVQ(OneClassMixin, GLVQ):
    def __init__(self, hparams, **kwargs):
        distance_fn = kwargs.get("distance_fn", squared_euclidean_distance)
        x_train = kwargs.get("theta_initializer", None)

        super().__init__(hparams, distance_fn=distance_fn, **kwargs)

        if x_train is None:
            raise NotImplementedError("No default theta initializer")

        theta_init = ThetaInitializer(x_train, self)
        self.init_variant(theta_init=theta_init)
        self.init_params()


class OneClassGMLVQ(OneClassMixin, GMLVQ):
    def __init__(self, hparams, **kwargs):
        super().__init__(hparams, **kwargs)

        x_train = kwargs.pop("theta_initializer")

        loss = kwargs.pop("loss", occ_csi_soft_loss2)
        theta_trainable = kwargs.pop("theta_trainable", True)

        if x_train is None:
            raise NotImplementedError("No default theta initializer")

        theta_init = ThetaInitializer(x_train, self)
        self.init_variant(
            theta_init=theta_init,
            loss=loss,
            theta_trainable=theta_trainable,
        )
        self.init_params()

    def lambda_matrix(self):
        lam = self._omega @ self._omega.T
        return lam.detach().cpu()

    def predict_latent(self, x, map_protos=True):
        """Predict `x` assuming it is already embedded in the latent space.

        Only the prototypes are embedded in the latent space using the
        backbone.

        """
        self.eval()
        with torch.no_grad():
            protos, plabels = self.proto_layer()
            if map_protos:
                protos = self.backbone(protos)
            d = squared_euclidean_distance(x, protos)
            y_pred = self.wtac(d, plabels, self._theta)
        return y_pred


class OneClassLGMLVQ(OneClassMixin, LGMLVQ):
    def __init__(self, hparams, **kwargs):
        distance_fn = kwargs.pop("distance_fn", lomega_distance)
        x_train = kwargs.pop("theta_initializer")
        super().__init__(hparams, distance_fn=distance_fn, **kwargs)

        # Re-register `_omega` to override the one from the super class.
        omega = torch.randn(
            self.num_prototypes,
            self.hparams.input_dim,
            self.hparams.latent_dim,
            device=self.device,
        )
        self.register_parameter("_omega", Parameter(omega))

        theta_init = ThetaInitializer(x_train, self.compute_distances,
                                      self.proto_layer, self.device)
        self.init_variant(theta_init=theta_init)
        self.init_params()

    """
    def lambda_matrix(self):
        lam = self._omega @ self._omega.T
        return lam.detach().cpu()
    """

    def predict_latent(self, x, map_protos=True):
        """Predict `x` assuming it is already embedded in the latent space.

        Only the prototypes are embedded in the latent space using the
        backbone.

        """
        self.eval()
        with torch.no_grad():
            protos, plabels = self.proto_layer()
            if map_protos:
                protos = self.backbone(protos)
            d = squared_euclidean_distance(x, protos)
            y_pred = self.wtac(d, plabels, self._theta)
        return y_pred

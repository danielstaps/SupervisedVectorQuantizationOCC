""" One Class Classifier based on GLVQ framework """

import torch
import numpy as np
from torch.nn.parameter import Parameter

from prototorch.models.glvq import GLVQ, SiameseGLVQ, GMLVQ, LGMLVQ
from .functions.competitions import wtac_thresh
from .functions.losses import one_class_classifier_loss, one_class_classifier_triplet_loss, occ_loss, occ_mitRonny, occ_studentT_loss, occ_studentT_loss_v2
from .functions.losses_csi import occ_csi_soft_loss, occ_brier_score, occ_heidke_skill_score, occ_brier_score2, occ_csi_soft_loss2
from prototorch.nn import LambdaLayer

from prototorch.core.distances import (
    lomega_distance,
    omega_distance,
    squared_euclidean_distance,
    euclidean_distance,
)

from torch.optim.lr_scheduler import ExponentialLR


class ThetaInitializer():
    def __init__(self, x_train, num_thetas=None):
        if num_thetas:
            self.num_thetas = num_thetas 
        if type(x_train) == np.ndarray:
            x_train = torch.Tensor(x_train)
        d = self.compute_distances(x_train).detach().numpy()
        self.theta_init = float(np.mean(d, axis=0))
        print("d median",theta_init)

    def theta(self,):
        print("label_shape:",self.proto_layer.labels.shape)
        otheta = torch.full(self.proto_layer.labels.shape, self.theta_init, device=self.device, requires_grad=theta_trainable)
        theta = torch.abs(otheta)
        return theta



class OneClassMixin():
    def init_variant(self,):
        # Additional parameters
        #theta = ThetaInitializerPerPrototype(num_thetas=self.proto_layer.labels.shape, theta=10.).generate()
        #theta = torch.randn(self.proto_layer.labels.shape,
        #                    device=self.device)
        theta = torch.full(self.proto_layer.labels.shape, theta_init, device=self.device)
        theta = torch.pow(theta, 2)
        self.register_parameter("_theta", Parameter(theta))
        #self.loss = LambdaLayer(one_class_classifier_loss)
        self.loss = LambdaLayer(occ_mitRonny)
        self.wtac = wtac_thresh # Vorschlag, denn auch beim SMI-GMLVQ wird die wtac leicht abgeändert

    def init_variant_2(self, theta_init=theta_obj, theta_trainable=True):
        print("label_shape:",self.proto_layer.labels.shape)
        otheta = torch.full(self.proto_layer.labels.shape, theta_init, device=self.device, requires_grad=theta_trainable)
        theta = torch.abs(otheta)
        self.register_parameter("_theta", Parameter(theta_obj.theta))
        #self.loss = LambdaLayer(occ_studentT_loss)
        #self.loss = LambdaLayer(occ_csi_soft_loss)
        #self.loss = LambdaLayer(occ_csi_soft_loss2)
        self.loss = LambdaLayer(occ_brier_score)
        #self.loss = LambdaLayer(occ_brier_score2)
        #self.loss = LambdaLayer(occ_heidke_skill_score)
        self.wtac = wtac_thresh # Vorschlag, denn auch beim SMI-GMLVQ wird die wtac leicht abgeändert   

    def init_params(self,):
        self.lr_scheduler = ExponentialLR
        self.lr_scheduler_kwargs = dict(gamma=0.99, verbose=False)
    
    def shared_step(self, batch, batch_idx, optimizer_idx=None):
        x, y = batch
        out = self.compute_distances(x)
        plabels = self.proto_layer.labels
        loss = self.loss(out, y, prototype_labels=plabels, theta_boundary=self._theta, device=self.device)
        #mu = self.loss(out, y, prototype_labels=plabels, theta_boundary=self._theta)
        #batch_loss = self.transfer_layer(mu, beta=self.hparams.transfer_beta)
        #loss = batch_loss.sum(dim=0)
        return out, loss

    @property
    def theta_boundary(self):
        return self._theta.detach().cpu()

    def predict_from_distances(self, distances):
        with torch.no_grad():
            plabels = self.proto_layer.labels
            y_pred = self.wtac(distances, plabels, self._theta)
        return y_pred


class OneClassGLVQv2(OneClassMixin, GLVQ):
    def __init__(self, hparams, **kwargs):
        distance_fn = kwargs.pop("distance_fn", squared_euclidean_distance)
        super().__init__(hparams, distance_fn=distance_fn, **kwargs)
        #super().__init__(hparams, **kwargs)

        self.init_variant_2(theta_init=ThetaInitializer(x_train))
        self.init_params()



class OneClassGMLVQv2(OneClassMixin, GMLVQ):
    def __init__(self, hparams, **kwargs):
        distance_fn = kwargs.pop("distance_fn", omega_distance)
        x_train = kwargs.pop("theta_initializer")
        super().__init__(hparams, distance_fn=distance_fn, **kwargs)
        #super().__init__(hparams, **kwargs)

        self.init_variant_2(theta_init=ThetaInitializer(x_train))
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



class OneClassLGMLVQv2(OneClassMixin, LGMLVQ):
    def __init__(self, hparams, **kwargs):
        distance_fn = kwargs.pop("distance_fn", lomega_distance)
        super().__init__(hparams, distance_fn=distance_fn, **kwargs)
 
        # Re-register `_omega` to override the one from the super class.
        omega = torch.randn(
            self.num_prototypes,
            self.hparams.input_dim,
            self.hparams.latent_dim,
            device=self.device,
        )
        self.register_parameter("_omega", Parameter(omega))

        #super().__init__(hparams, **kwargs)
        self.init_variant_2(theta_init=ThetaInitializer(x_train))
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



class OneClassGLVQ(OneClassMixin, GLVQ):
    def __init__(self, hparams, **kwargs):
        distance_fn = kwargs.pop("distance_fn", squared_euclidean_distance)
        super().__init__(hparams, distance_fn=distance_fn, **kwargs)
        #super().__init__(hparams, **kwargs)
        self.init_variant()
        self.init_params()



class OneClassGMLVQ(OneClassMixin, GMLVQ):
    def __init__(self, hparams, **kwargs):
        distance_fn = kwargs.pop("distance_fn", omega_distance)
        super().__init__(hparams, distance_fn=distance_fn, **kwargs)
        #super().__init__(hparams, **kwargs)
        self.init_variant()
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
        super().__init__(hparams, distance_fn=distance_fn, **kwargs)
 
        # Re-register `_omega` to override the one from the super class.
        omega = torch.randn(
            self.num_prototypes,
            self.hparams.input_dim,
            self.hparams.latent_dim,
            device=self.device,
        )
        self.register_parameter("_omega", Parameter(omega))

        #super().__init__(hparams, **kwargs)
        self.init_variant()
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


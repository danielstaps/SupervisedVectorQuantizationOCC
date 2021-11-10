""" One Class Classifier based on GLVQ framework """

import torch
from torch.nn.parameter import Parameter

from prototorch.models.glvq import GLVQ, SiameseGLVQ, GMLVQ
from .functions.competitions import wtac_thresh
from .functions.losses import one_class_classifier_loss, one_class_classifier_triplet_loss, occ_loss, occ_mitRonny, occ_studentT_loss
from prototorch.nn import LambdaLayer

from prototorch.core.distances import (
    lomega_distance,
    omega_distance,
    squared_euclidean_distance,
    euclidean_distance,
)



class ThetaInitializerPerPrototype():
    def __init__(self, num_thetas, theta=0.1):
        self.theta = theta
        self.num_thetas = num_thetas 

    def generate(self, ):
        return torch.full((self.num_thetas,1), self.theta, requires_grad=True)



class OneClassMixin():
    def init_variant_1(self,):
        # Additional parameters
        #theta = ThetaInitializerPerPrototype(num_thetas=self.proto_layer.labels.shape, theta=10.).generate()
        #theta = torch.randn(self.proto_layer.labels.shape,
        #                    device=self.device)
        theta = torch.full(self.proto_layer.labels.shape, 0.00001, device=self.device)
        theta = torch.pow(theta, 2)
        self.register_parameter("_theta", Parameter(theta))
        #self.loss = LambdaLayer(one_class_classifier_loss)
        self.loss = LambdaLayer(occ_mitRonny)
        self.wtac = wtac_thresh # Vorschlag, denn auch beim SMI-GMLVQ wird die wtac leicht abgeändert   

    def init_variant_2(self,):
        print(self.proto_layer.labels.shape)
        theta = torch.full(self.proto_layer.labels.shape, 0.2, device=self.device)
        theta = torch.pow(theta, 2)
        self.register_parameter("_theta", Parameter(theta))
        self.loss = LambdaLayer(occ_studentT_loss)
        self.wtac = wtac_thresh # Vorschlag, denn auch beim SMI-GMLVQ wird die wtac leicht abgeändert   
   
    def shared_step(self, batch, batch_idx, optimizer_idx=None):
        x, y = batch
        out = self.compute_distances(x)
        plabels = self.proto_layer.labels
        loss = self.loss(out, y, prototype_labels=plabels, theta_boundary=self._theta)
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
        self.init_variant_2()


class OneClassGLVQ(OneClassMixin, GLVQ):
    def __init__(self, hparams, **kwargs):
        distance_fn = kwargs.pop("distance_fn", squared_euclidean_distance)
        super().__init__(hparams, distance_fn=distance_fn, **kwargs)
        #super().__init__(hparams, **kwargs)
        self.init_variant_1()



class OneClassGMLVQ(OneClassMixin, GMLVQ):
    def __init__(self, hparams, **kwargs):
        distance_fn = kwargs.pop("distance_fn", omega_distance)
        super().__init__(hparams, distance_fn=distance_fn, **kwargs)
        #super().__init__(hparams, **kwargs)
        self.init_variant_1()

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


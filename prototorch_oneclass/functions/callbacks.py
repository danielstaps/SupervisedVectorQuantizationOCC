import torch
from prototorch.core.distances import squared_euclidean_distance
from pytorch_lightning.callbacks import Callback


class ThetaCallback(Callback):
    def __init__(self, train_ds):
        self.train_ds = train_ds

    def on_train_batch_end(self, trainer, pl_module, *args, **kwargs):
        state_dict = pl_module.state_dict()
        """
        if min(actual_theta) <= 1e-3:
            print("Theta has value", actual_theta)
            print(
                "Warning: Theta Minimum is low! Please consider increasing LATENT_DIM or decreasing PROTOTYPES Parameters."
            )
        if min(actual_theta) <= 1e-5:
            #should_stop = True
            #should_stop = trainer.training_type_plugin.reduce_boolean_decision(
            #    should_stop)
            #trainer.should_stop = trainer.should_stop or should_stop
            print(
                f"Attention! Theta Minimum was to low! Theta={min(actual_theta)} Please consider increasing LATENT_DIM or decreasing PROTOTYPES Parameters."
            )
        """
        classes = torch.unique(pl_module.prototype_labels)
        min_max = []
        for i in classes:
            x = self.train_ds.data[self.train_ds.target == i, :]
            if hasattr(pl_module, '_omega'):
                d_class = pl_module.distance_layer(x, x, pl_module._omega)
            else:
                d_class = pl_module.distance_layer(x, x)
            min_max.append([
                #torch.amin(d_class[d_class != 0]) * 0.5,
                torch.quantile(d_class, 0.2) * 0.5,
                torch.amax(d_class) * 0.5,
            ])
        for i in classes:
            ii = pl_module.prototype_labels == i
            for j, e in enumerate(ii):
                if e:
                    state_dict['_theta'][j] = torch.clip(
                        state_dict['_theta'][j],
                        min=min_max[i][0],
                        max=min_max[i][1])
        pl_module.load_state_dict(state_dict)


class SigmaCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module) -> None:
        state_dict = pl_module.state_dict()
        #state_dict['_sigma'] -= torch.Tensor([0.8 / trainer.max_epochs])
        #state_dict['_sigma'] *= 0.9991
        state_dict['_sigma'] = torch.exp(torch.Tensor([-trainer.current_epoch / trainer.max_epochs]))
        pl_module.load_state_dict(state_dict)

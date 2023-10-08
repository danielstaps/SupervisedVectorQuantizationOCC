import torch
from pytorch_lightning.callbacks import Callback


class ThetaCallback(Callback):

    def __init__(self, train_ds):
        self.train_ds = train_ds

    def on_train_batch_end(self, trainer, pl_module, *args, **kwargs):
        state_dict = pl_module.state_dict()
        classes = torch.unique(pl_module.prototype_labels)
        min_max = []
        for i in classes:
            x = self.train_ds.data[self.train_ds.target == i, :]
            x = x[:2500, :]
            if hasattr(pl_module, '_omega'):
                d_class = pl_module.distance_layer(x, x, pl_module._omega)
            elif hasattr(pl_module, '_scale'):
                d_class = pl_module.distance_layer(x, x, pl_module._scale)
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


class ScaleCallback(Callback):

    def on_train_batch_end(self, trainer, pl_module, *args, **kwargs):
        state_dict = pl_module.state_dict()
        if '_scale' in state_dict:
            state_dict['_scale'] = torch.clamp(state_dict['_scale'],
                                               torch.Tensor([0.001]),
                                               torch.Tensor([50.0]))
        pl_module.load_state_dict(state_dict)


class DynamicCallback(Callback):

    def sigmoid_sigma(self, max_e, current_e):
        return 0.0001 + 0.01 * (1 / (1 + torch.exp(
            torch.Tensor([(current_e - max_e / 1.8) * 6. / max_e]))))

    def sigmoid_alpha(self, max_e, current_e):
        return 0.5 + 0.5 * (1 / (1 + torch.exp(
            torch.Tensor([(current_e - max_e / 4.) * 20. / max_e]))))

    def lin_pieces(self, max_e, current_e):
        break_point = 0.25
        if current_e >= break_point * max_e:
            return torch.Tensor([0.27])
        else:
            return torch.Tensor(
                [0.5 - ((0.5 - 0.27) / (max_e * break_point)) * current_e])

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        state_dict = pl_module.state_dict()
        """
        Probability soft to sharp
        """
        state_dict['_sigma'] = self.sigmoid_sigma(trainer.max_epochs,
                                                  trainer.current_epoch)
        """
        Neural Gas Parameters
        """
        state_dict['_ng_lambda'] = self.lin_pieces(trainer.max_epochs,
                                                   trainer.current_epoch)
        """
        Loss weighting
        """
        state_dict['_alpha'] = self.sigmoid_alpha(trainer.max_epochs,
                                                  trainer.current_epoch)

        pl_module.load_state_dict(state_dict)

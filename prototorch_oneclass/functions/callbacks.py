from pytorch_lightning.callbacks import Callback
from torch import Tensor


class ThetaCallback(Callback):
    def on_train_batch_end(self, trainer, pl_module, *args, **kwargs):
        actual_theta = pl_module.state_dict()['_theta']
        if min(actual_theta) <= 1e-3:
            print("Theta has value", actual_theta)
            print(
                "Warning: Theta Minimum is low! Please consider increasing LATENT_DIM or decreasing PROTOTYPES Parameters."
            )
        if min(actual_theta) <= 1e-5:
            should_stop = True
            should_stop = trainer.training_type_plugin.reduce_boolean_decision(
                should_stop)
            trainer.should_stop = trainer.should_stop or should_stop
            print(
                f"Attention! Theta Minimum was to low! Theta={min(actual_theta)} Please consider increasing LATENT_DIM or decreasing PROTOTYPES Parameters."
            )


class SigmaCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module) -> None:
        max_epochs = trainer.max_epochs
        state_dict = pl_module.state_dict()
        state_dict['_sigma'] -= Tensor([0.99 / max_epochs])
        pl_module.load_state_dict(state_dict)

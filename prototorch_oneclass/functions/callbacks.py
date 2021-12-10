from pytorch_lightning.callbacks import Callback


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

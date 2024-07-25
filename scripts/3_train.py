import os
import torch

# Custom Imports
from core.models import UnetWrapper

synth_params = {
    "parenchyma": {
        "nb_classes": 5,
        "shape": 10
    },
    "gamma": [0.2, 2],
    "z_decay": [32],
    "speckle": [0.2, 0.8],
    "imax": 0.80,
    "imin": 0.01,
    "vessel_texture": True,
    "spheres": True,
    "slabwise_banding": True,
    "dc_offset": True
}

if __name__ == "__main__":
    ##################
    version_n = 1
    ##################

    data_experiment_number = 2
    print(f'Using data from experiment {data_experiment_number}')

    # New unet
    unet = UnetWrapper(
        version_n=version_n,
        synth_params=synth_params,
        model_dir='models',
        learning_rate=1e-3
        )

    unet.new(
        nb_levels=4,
        nb_features=[32, 64, 128, 256],
        dropout=0,
        augmentation=True)

    n_vol = 1000
    train_to_val = 0.8
    n_steps = 1e5
    n_gpus = 1
    accum_grad = 1
    batch_size = 1

    n_train = n_vol*train_to_val
    batch_sz_eff = batch_size * n_gpus * accum_grad
    epochs = int((n_steps * batch_sz_eff) // n_train)
    print(f'Training for {epochs} epochs')
    unet.train_it(
        data_experiment_number=data_experiment_number,
        epochs=epochs,
        batch_size=batch_size,
        accumulate_gradient_n_batches=accum_grad,
        num_workers=1,
        train_to_val=train_to_val
    )

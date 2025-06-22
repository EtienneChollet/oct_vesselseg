import os
from typing import Literal

from cyclopts import App
import warnings

warnings.simplefilter("ignore")

app = App()


@app.command()
def configure():
    """
    Configures ~/.bashrc for oct_vesselseg project by setting \\
    OCT_VESSELSEG_BASE_DIR environment variable to specified directory.
    """
    variable_name = 'OCT_VESSELSEG_BASE_DIR'
    bashrc_path = os.path.expanduser("~/.bashrc")
    # Collect user input
    print('Please enter the FULL path to the output directory for '
          'oct_vesselseg')
    oct_vesselseg_base_dir = input('> ').rstrip('/')

    # Check if the base directory exists, create it if it doesn't
    if not os.path.exists(oct_vesselseg_base_dir):
        try:
            os.makedirs(oct_vesselseg_base_dir)
            print(
                f"Directory '{oct_vesselseg_base_dir}' created successfully.")
        except OSError as e:
            print(
                f"Failed to create directory '{oct_vesselseg_base_dir}': {e}")
            return

    export_command = f'export {variable_name}="{oct_vesselseg_base_dir}"\n'

    # Check if the variable already exists in .bashrc
    if os.path.exists(bashrc_path):
        with open(bashrc_path, 'r') as file:
            lines = file.readlines()

        # Update the variable if it exists
        for i, line in enumerate(lines):
            if line.startswith(f'export {variable_name}='):
                lines[i] = export_command
                break
        else:
            # If the variable does not exist, append it
            lines.append(export_command)

        with open(bashrc_path, 'w') as file:
            file.writelines(lines)
    else:
        # If the .bashrc file doesn't exist, create it and add the variable
        with open(bashrc_path, 'w') as file:
            file.write(export_command)

    print("\nConfiguration successful. "
          "Please run `source ~/.bashrc` to apply the changes, "
          "or restart your terminal.")


@app.command()
def vesselsynth(data_experiment_n: int = 1,
                shape: tuple[int, int, int] = [128, 128, 128],
                n_samples: int = 1000,
                voxel_size: float = 0.02,
                tree_levels: tuple[int, int] = [1, 4],
                tree_density: tuple[float, float] = [0.1, 0.2],
                tree_root_radius: tuple[float, float] = [0.1, 0.15],
                branch_tortuosity: tuple[int, int] = [1, 5],
                branch_radius_ratio: tuple[float, float] = [0.25, 1],
                branch_radius_change: tuple[float, float] = [0.9, 1.1],
                branch_children: tuple[int, int] = [1, 4],
                device: str = 'cuda'
                ):
    """
    Synthesize volumetric vessel labels with morphological parameters by \\
    sampling probability density functions.

    Parameters
    ----------
    data_experiment_n : int
        Data experiment number for saving volumetric data and synthesis
        description.
    shape : list
        Shape of the synthetic volume in voxels (x, y, z).
    n_samples : int
        Number of vessel label samples to make.
    voxel_size : float
        Resolution of the synthetic volume in mm^3 per voxel.
    tree_levels : list[int]
        Number of hierarchical levels in the vascular tree.
    tree_density : list[float]
        Density of the vascular tree structures per cubic mm (trees/mm^3).
    tree_root_radius : list[float]
        Sampler bounds for radius of vascular tree trunk in mm.
    branch_tortuosity : list[float]
        Sampler for tortuosity of vasculature (tortuosity ~= cord / length)
    branch_radius_ratio : list[float]
        Sampler bounds for the ratio of the radius of children vessels to the
        parent vessel.
    branch_radius_change : list[float]
        Sampler bounds for a multiplicative variation in radius along the legth
        of a vessel
    branch_children : list[int]
        Sampler bounds for the number of branches per tree.
    device : str
        Device to perform the computations on. Default is 'cuda'
    """
    import torch
    from oct_vesselseg.synth import (
        VesselSynthEngineWrapper, VesselSynthEngineOCT)
    from synthspline.random import Uniform, RandInt
    synth_params = {
        'shape': shape,
        'voxel_size': voxel_size,
        'nb_levels': RandInt(*tree_levels),
        'tree_density': Uniform(*tree_density),
        'tortuosity': Uniform(*branch_tortuosity),
        'radius': Uniform(*tree_root_radius),
        'radius_ratio': Uniform(*branch_radius_ratio),
        'radius_change': Uniform(*branch_radius_change),
        'nb_children': RandInt(*branch_children),
        'device': device
        }

    torch.no_grad()
    synth_engine = VesselSynthEngineOCT(**synth_params)
    VesselSynthEngineWrapper(
        experiment_number=data_experiment_n,
        synth_engine=synth_engine,
        n_volumes=n_samples,
        ).synth()


@app.command()
def imagesynth(data_experiment_n: int = 1,
               n_samples: int = 10,
               parenchyma_classes: int = 5,
               parenchyma_shape: int = 10,
               vessel_intensity: tuple[float, float] = [0.01, 0.8],
               vessel_texture: bool = True,
               vessel_random_ablation: bool = True,
               image_gamma: tuple[float, float] = [0.2, 2],
               image_z_decay: int = 32,
               image_speckle: tuple[float, float] = [0.2, 0.8],
               image_spheres: bool = True,
               image_banding: bool = True,
               image_dc_offset: bool = True
               ):
    """
    Synthesize OCT images with optional noise/artifact models and save in \\
    synthetic experiment directory.

    Parameters
    ----------
    data_experiment_n : int
        Data experiment number for loading and volumetric data.
    n_samples : int
        Number of samples to synthesize and save to data experiment.
    parenchyma_classes : int
        Sampler upper bound for number of classes of parenchyma/neural tissue.
    parenchyma_shape : int
        Sampler upper bound for number of control points in a given
        parenchyma class.
    image_gamma : list
        Sampler bounds for non-linear contrast adjustment/stretch.
        Larger values increase contrast whereas lower values decrease contrast.
    image_z_decay : list
        Z decay upper bound
    image_speckle : list
        Sampler bounds for speckle noise parameters.
    vessel_intensity : list
        Sampler bounds for weighted blending of vessels onto parenchyma.
    vessel_texture : bool
        Apply intra-vascular textures/artifacts.
    vessel_random_ablation : bool
        Optionally ablate vessels randomly.
    image_spheres : bool
        Apply sphere artifacts to image.
    image_banding : bool
        Apply slabwise banding (z-decay) artifact to the image.
    image_dc_offset : bool
        Add a small value to the parenchyma tensor.
    """
    from oct_vesselseg.synth import ImageSynthEngineWrapper
    synth_params = {
        "parenchyma": {
            "nb_classes": parenchyma_classes,
            "shape": parenchyma_shape
        },
        "random_vessel_ablation": vessel_random_ablation,
        "gamma": image_gamma,
        "z_decay": [image_z_decay],
        "speckle": image_speckle,
        "imin": vessel_intensity[0],
        "imax": vessel_intensity[1],
        "vessel_texture": vessel_texture,
        "spheres": image_spheres,
        "slabwise_banding": image_banding,
        "dc_offset": image_dc_offset
    }

    vesselseg_outdir = os.getenv("OCT_VESSELSEG_BASE_DIR")
    synth = ImageSynthEngineWrapper(
        exp_path=(f"{vesselseg_outdir}/synthetic_data/"
                  f"exp{data_experiment_n:04}"),
        synth_params=synth_params,
        save_nifti=True,
        save_fig=True
        )
    for i in range(n_samples):
        synth[i]


@app.command()
def train(
    model_version_n: int = 1,
    model_dir: str = 'models',
    model_levels: int = 4,
    model_features: tuple[int] = [32, 64, 128, 256],
    training_lr: float = 1e-3,
    training_train_to_val: float = 0.8,
    training_steps: int = 1e5,
    training_batch_size: int = 1,
    synth_data_experiment_n: int = 1,
    synth_samples: int = 1000,
    synth_parenchyma_classes: int = 5,
    synth_parenchyma_shape: int = 10,
    synth_image_gamma: tuple[float, float] = [0.2, 2],
    synth_image_z_decay: int = 32,
    synth_image_speckle: tuple[float, float] = [0.2, 0.8],
    synth_vessel_intensity: tuple[float, float] = [0.01, 0.8],
    synth_vessel_texture: bool = True,
    synth_image_spheres: bool = True,
    synth_image_banding: bool = True,
    synth_image_dc_offset: bool = True,
    synth_random_vessel_ablation: bool = True,
        ):
    """
    Train a Unet with specified model and imagesynth parameters.

    Parameters
    ----------
    model_version_n : int
        Version number of the model to train.
    model_dir : str
        directory where all model versions/experimental runs are stored.
    model_levels : int
        Number of levels (encoding/decoding block pairs) of the model.
    model_features : list
        List of integers defining the number of features in each corresponding
        layer of the model.
    training_lr : float
        Global learning rate that will be used for the majority of training
        (after lr warmup and before lr cooldown)
    training_train_to_val : float
        Ratio of training samples to validation samples (train:val).
    training_steps : int
        Number of times to step the optimizer (update the model's parameters)
        based on gradients calculated in backward pass.
    training_batch_size : int
        Number of samples per batch.
    synth_data_experiment_n : int
        Data experiment number for loading and volumetric data and synthesis
        description.
    synth_samples : int
        Number of unique synthetic vessel label tensors to use.
    synth_parenchyma_classes : int
        Sampler upper bound for number of classes of parenchyma/neural tissue.
    synth_parenchyma_shape : int
        Sampler upper bound for number of control points in a given
        parenchyma class.
    synth_image_gamma : list
        Mean and standard deviation of multiplicative gamma noise.
    synth_image_z_decay : list
        Z decay upper bound
    synth_image_speckle : list
        Sampler bounds for speckle noise parameters.
    synth_vessel_intensity : list
        Sampler bounds for weighted blending of vessels onto parenchyma.
    synth_vessel_texture : bool
        Apply intra-vascular textures/artifacts.
    synth_image_spheres : bool
        Apply sphere artifacts to image.
    synth_image_banding : bool
        Apply slabwise banding (z-decay) artifact to the image.
    synth_image_dc_offset : bool
        Add a small value to the parenchyma tensor.
    synth_random_vessel_ablation : bool
        Optionally hide vessels randomly.
    """
    from oct_vesselseg.models import UnetWrapper
    synth_params = {
        "parenchyma": {
            "nb_classes": synth_parenchyma_classes,
            "shape": synth_parenchyma_shape
        },
        "gamma": synth_image_gamma,
        "z_decay": [synth_image_z_decay],
        "speckle": synth_image_speckle,
        "imin": synth_vessel_intensity[0],
        "imax": synth_vessel_intensity[1],
        "vessel_texture": synth_vessel_texture,
        "spheres": synth_image_spheres,
        "slabwise_banding": synth_image_banding,
        "dc_offset": synth_image_dc_offset,
        "random_vessel_ablation": synth_random_vessel_ablation
        }

    # Init a new Unet
    unet = UnetWrapper(
        version_n=model_version_n,
        synth_params=synth_params,
        model_dir=model_dir,
        learning_rate=training_lr
        )
    unet.new(
        nb_levels=model_levels,
        nb_features=model_features,
        dropout=0,
        augmentation=True)

    unet.train_it(
        synth_data_experiment_n=synth_data_experiment_n,
        synth_samples=synth_samples,
        training_steps=training_steps,
        train_to_val=training_train_to_val,
        batch_size=training_batch_size,
    )


@app.command()
def test(in_path: str, model_version_n: int = 1, model_dir: str = 'models',
         patch_size: int = 128, redundancy: int = 3, checkpoint: str = 'best',
         padding_method: str = 'reflect', normalize_patches: bool = True
         ):
    """
    Test a trained Unet.

    Parameters
    ----------
    in_path : str
        Path to stitched OCT mus data in NIfTI format. Can test on many
        different input files by seperating paths by commas.
    model_version_n : int
        Version number of the model to test.
    model_dir : str
        Directory within output folder containing model versions.
    patch_size : int
        Size of Unet in each dimension.
    redundancy: int
        Redundancy factor for prediction overlap (default: 3).
    checkpoint : str
        Which checkpoint to load weights from. {'best', 'last'}.
    padding_method : str
        Method to pad the input tensor. {'reflect', 'replicate', 'constant'}
    normalize_patches : bool
        Optionally normalize each patch before prediction

    """
    import gc
    import time
    import torch
    from oct_vesselseg.models import UnetWrapper
    from oct_vesselseg.data import RealOctPredict, RealOctConfig

    # Starting timer
    t1 = time.time()
    if ',' in in_path:
        in_path = in_path.split(',')
    else:
        in_path = [in_path] if isinstance(in_path, str) else in_path
    print(in_path)

    # Make the prediction without gradient computations
    with torch.no_grad():
        for path in in_path:
            # Init the unet
            unet = UnetWrapper(
                version_n=model_version_n,
                model_dir=model_dir,
                device='cuda'
            )

            # Loading model weights and setting to test mode
            unet.load(type=checkpoint, mode='test')
            # Configuring prediction
            oct_config = RealOctConfig(
                input=path,
                patch_size=patch_size,
                redundancy=redundancy,
                # trainee=unet.trainee,
                pad_it=True,
                padding_method=padding_method,
                normalize=normalize_patches
            )

            prediction = RealOctPredict(oct_config, trainee=unet.trainee)
            prediction.predict_on_all()
            out_path = f"{unet.version_path}/predictions"
            prediction.save_prediction(dir=out_path)
            t2 = time.time()
            print(f"Process took {round((t2-t1)/60, 2)} min")
            print('#' * 30, '\n')
            del unet
            del prediction
            torch.cuda.empty_cache()
            gc.collect()

@app.command()
def client(
    in_path: str,
    out_path: str | None = 'oct_vesselseg_output.nii.gz',
    patch_size: int = 128, 
    redundancy: int = 3,
    pad_it: bool = True,
    padding_method: Literal['reflect', 'replicate', 'constant'] = 'reflect',
    normalize_patches: bool = True,
    api_url: str = 'http://127.0.0.1:8000/predict',
) -> None:
    """
    Request OCT Volume segmentation (on the `darwin`) from segmentation API.

    Parameters
    ----------
    in_path : str
        Absolute path on the server to the NIfTI file you wish to segment.
    out_path : str | 'oct_vesselseg_output.nii.gz', optional
        Local path to save the returned bytes. If None, will print
        first 100 bytes to stdout.
    patch_size : int
        Size of Unet in each dimension.
    redundancy: int
        Redundancy factor for prediction overlap (default: 3).
    padding_method : str
        Method to pad the input tensor.
    normalize_patches : bool
        Optionally normalize each patch before prediction
    api_url : str, optional
        Full URL to the `/predict` endpoint, e.g.
        "http://127.0.0.1:8000/predict".
    """
    import requests

    data = {
        "in_path": in_path,
        "out_path": out_path,
        "patch_size": patch_size,
        "redundancy": redundancy,
        "pad_it": str(pad_it).lower(),
        "padding_method": padding_method,
        "normalize_patches": str(normalize_patches).lower(),
    }

    with requests.post(api_url, data=data, stream=True) as resp:
        resp.raise_for_status()

        for chunk in resp.iter_content(chunk_size=8192):
            try:
                text = chunk.decode("utf-8")
                print(text, end="")
            except:
                pass


if __name__ == '__main__':
    app()

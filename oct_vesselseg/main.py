from cyclopts import App
import warnings

warnings.simplefilter("ignore")

app = App()


@app.command()
def configure():
    """
    Configures ~/.bashrc for oct_vesselseg project by setting OCT_VESSELSEG_BASE_DIR environment variable to specified directory.
    """
    import os
    variable_name = 'OCT_VESSELSEG_BASE_DIR'
    bashrc_path = os.path.expanduser("~/.bashrc")
    # Collect user input
    print('Please enter the path to the output directory for oct_vesselseg')
    oct_vesselseg_base_dir = input('> ')

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

    # Optionally, source the .bashrc file (applies the changes immediately in
    # current session)
    os.system(f'source {bashrc_path}')


@app.command()
def vesselsynth(exp: int = 1,
                shape: tuple[int, int, int] = [128, 128, 128],
                voxel_size: float = 0.02,
                levels: tuple[int, int] = [1, 4],
                density: tuple[float, float] = [0.1, 0.2],
                tortuosity: tuple[int, int] = [1, 5],
                root_radius: tuple[float, float] = [0.1, 0.15],
                radius_ratio: tuple[float, float] = [0.25, 1],
                radius_change: tuple[float, float] = [0.9, 1.1],
                children: tuple[int, int] = [1, 4],
                device: str = 'cuda'
                ):
    """
    Synthesize volumetric vessel labels with morphological parameters by sampling probability density functions.

    Parameters
    ----------
    exp : int
        Data experiment number for saving volumetric data and synthesis
        description.
    shape : list
        Shape of the synthetic volume in voxels (x, y, z).
    voxel_size : float
        Resolution of the synthetic volume in mm^3 per voxel.
    levels : list[int]
        Number of hierarchical levels in the vascular tree.
    density : list[float]
        Density of the vascular tree structures per cubic mm (trees/mm^3).
    tortuosity : list[float]
        Sampler for tortuosity of vasculature (tortuosity ~= cord / length)
    root_radius : list[float]
        Sampler bounds for radius of vascular tree trunk in mm.
    radius_ratio : list[float]
        Sampler bounds for the ratio of the radius of children vessels to the
        parent vessel.
    radius_change : list[float]
        Sampler bounds for a multiplicative variation in radius along the legth
        of a vessel
    children : list[int]
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
        'nb_levels': RandInt(*levels),
        'tree_density': Uniform(*density),
        'tortuosity': Uniform(*tortuosity),
        'radius': Uniform(*root_radius),
        'radius_ratio': Uniform(*radius_ratio),
        'radius_change': Uniform(*radius_change),
        'nb_children': RandInt(*children),
        'device': device
        }

    torch.no_grad()
    synth_engine = VesselSynthEngineOCT(**synth_params)
    VesselSynthEngineWrapper(
        experiment_number=exp,
        synth_engine=synth_engine,
        ).synth()


@app.command()
def imagesynth(exp: int = 1,
               n_samples: int = 10,
               parenchyma_classes: int = 5,
               parenchyma_shape: int = 10,
               gamma: tuple[float, float] = [0.2, 2],
               z_decay: int = 32,
               speckle: tuple[float, float] = [0.2, 0.8],
               vessel_intensity: tuple[float, float] = [0.01, 0.8],
               vessel_texture: bool = True,
               spheres: bool = True,
               slabwise_banding: bool = True,
               dc_offset: bool = True
               ):
    """
    Synthesize OCT images with optional noise/artifact models and save in synthetic experiment directory.

    Parameters
    ----------
    exp : int
        Data experiment number for loading and volumetric data and synthesis
        description.
    n_samples : int
        Number of samples to synthesize and save to data experiment.
    parenchyma_classes : int
        Sampler upper bound for number of classes of parenchyma/neural tissue.
    parenchyma_shape : int
        Sampler upper bound for number of control points in a given
        parenchyma class.
    gamma : list
        Sampler bounds for non-linear contrast adjustment/stretch.
        Larger values increase contrast whereas lower values decrease contrast.
    z_decay : list
        Z decay upper bound
    speckle : list
        Sampler bounds for speckle noise parameters.
    vessel_intensity : list
        Sampler bounds for weighted blending of vessels onto parenchyma.
    vessel_texture : bool
        Apply intra-vascular textures/artifacts.
    spheres : bool
        Apply sphere artifacts to image.
    slabwise_banding : bool
        Apply slabwise banding (z-decay) artifact to the image.
    dc_offset : bool
        Add a small value to the parenchyma tensor.
    """
    from oct_vesselseg.synth import ImageSynthEngineWrapper
    synth_params = {
        "parenchyma": {
            "nb_classes": parenchyma_classes,
            "shape": parenchyma_shape
        },
        "gamma": gamma,
        "z_decay": [z_decay],
        "speckle": speckle,
        "imax": vessel_intensity[0],
        "imin": vessel_intensity[1],
        "vessel_texture": vessel_texture,
        "spheres": spheres,
        "slabwise_banding": slabwise_banding,
        "dc_offset": dc_offset
    }
    synth = ImageSynthEngineWrapper(
        exp_path=f"output/synthetic_data/{exp:04}",
        synth_params=synth_params,
        save_nifti=True,
        save_fig=True
        )
    for i in range(n_samples):
        synth[i]


@app.command()
def train(model_version: int = 1,
          model_dir: str = 'models',
          lr: float = 1e-3,
          model_levels: int = 4,
          model_features: tuple[int] = [32, 64, 128, 256],
          n_volumes: int = 1000,
          train_to_val: float = 0.8,
          n_steps: int = 1e5,
          batch_size: int = 1,
          exp: int = 1,
          parenchyma_classes: int = 5,
          parenchyma_shape: int = 10,
          gamma: tuple[float, float] = [0.2, 2],
          z_decay: int = 32,
          speckle: tuple[float, float] = [0.2, 0.8],
          vessel_intensity: tuple[float, float] = [0.01, 0.8],
          vessel_texture: bool = True,
          spheres: bool = True,
          slabwise_banding: bool = True,
          dc_offset: bool = True
          ):
    """
    Train a Unet with specified model and imagesynth parameters.

    Parameters
    ----------
    model_version : int
        Version number of the model to train.
    model_dir : str
        Directory within output folder to save model versions.
    lr : float
        Learning rate of the main training phase (between warmup and cooldown)
    model_levels : int
        Number of levels (encoding and decoding blocks) of the model.
    model_features : list
        List of number of features within the corresponging level of the model.
    train_to_val : float
        Ratio of training data to validation data.
    n_steps : int
        Number of steps the model will perform.
    batch_size : int
        Number of samples per batch.
    exp : int
        Data experiment number for loading and volumetric data and synthesis
        description.
    parenchyma_classes : int
        Sampler upper bound for number of classes of parenchyma/neural tissue.
    parenchyma_shape : int
        Sampler upper bound for number of control points in a given
        parenchyma class.
    gamma : list
        Sampler bounds for non-linear contrast adjustment/stretch.
        Larger values increase contrast whereas lower values decrease contrast.
    z_decay : list
        Z decay upper bound
    speckle : list
        Sampler bounds for speckle noise parameters.
    vessel_intensity : list
        Sampler bounds for weighted blending of vessels onto parenchyma.
    vessel_texture : bool
        Apply intra-vascular textures/artifacts.
    spheres : bool
        Apply sphere artifacts to image.
    slabwise_banding : bool
        Apply slabwise banding (z-decay) artifact to the image.
    dc_offset : bool
        Add a small value to the parenchyma tensor.
    """
    from oct_vesselseg.models import UnetWrapper
    synth_params = {
        "parenchyma": {
            "nb_classes": parenchyma_classes,
            "shape": parenchyma_shape
        },
        "gamma": gamma,
        "z_decay": [z_decay],
        "speckle": speckle,
        "imin": vessel_intensity[0],
        "imax": vessel_intensity[1],
        "vessel_texture": vessel_texture,
        "spheres": spheres,
        "slabwise_banding": slabwise_banding,
        "dc_offset": dc_offset
    }

    # Init a new Unet
    unet = UnetWrapper(
        version_n=model_version,
        synth_params=synth_params,
        model_dir=model_dir,
        learning_rate=lr
        )
    unet.new(
        nb_levels=model_levels,
        nb_features=model_features,
        dropout=0,
        augmentation=True)

    n_train = n_volumes*train_to_val
    epochs = int((n_steps * batch_size) // n_train)
    print(f'Training for {epochs} epochs')
    unet.train_it(
        data_experiment_number=exp,
        epochs=epochs,
        batch_size=batch_size,
        train_to_val=train_to_val
    )


@app.command()
def test(in_path: str, model_version: int = 1, model_dir: str = 'models',
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
    model_version : int
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
    from oct_vesselseg.data import RealOctPredict

    # Starting timer
    t1 = time.time()
    in_path = [in_path] if isinstance(in_path, str) else in_path
    print(in_path)
    # Make the prediction without gradient computations
    with torch.no_grad():
        for path in in_path:
            # Init the unet
            unet = UnetWrapper(
                version_n=model_version,
                model_dir=model_dir,
                device='cuda'
                )

            # Loading model weights and setting to test mode
            unet.load(type=checkpoint, mode='test')
            prediction = RealOctPredict(
                input=path,
                patch_size=patch_size,
                redundancy=redundancy,
                trainee=unet.trainee,
                pad_it=True,
                padding_method=padding_method,
                normalize_patches=normalize_patches,
                )

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


if __name__ == '__main__':
    app()

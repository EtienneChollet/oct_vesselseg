__all__ = [
    'VesselSynthEngineOCT',
    'VesselSynthEngineWrapper',
    'ImageSynthEngineOCT',
    'ImageSynthEngineWrapper',
    'VesselLabelDataset'
]

# Standard imports
import os
import json
import glob
import torch
import logging
import numpy as np
from torch import nn
from typing import Union, List, Dict
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

# Balbasty imports
import nibabel as nib
from synthspline.random import Uniform, RandInt, AnyVar
from synthspline.utils import default_affine
from synthspline.labelsynth import SynthSplineParameters, SynthSplineBlock
from cornucopia.labels import (
    RandomSmoothLabelMap,
    BernoulliDiskTransform)
from cornucopia import (
    RandomSlicewiseMulFieldTransform,
    RandomGammaTransform,
    RandomGammaNoiseTransform,
    RandomMulFieldTransform,
    RandomGaussianMixtureTransform,
    ElasticTransform,
    QuantileTransform
)
from cornucopia.random import Fixed, Normal

# Custom Imports
from oct_vesselseg.utils import PathTools

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] (%(levelname)s) ==> %(message)s',
    datefmt='%Y-%m-%d | %I:%M:%S %p'
)
# Init logger
logger = logging.getLogger(__name__)


vesselseg_outdir = os.getenv("OCT_VESSELSEG_BASE_DIR")


class VesselSynthEngineOCT(SynthSplineBlock):
    """Default parameters for SynthVesselOCT."""

    class defaults(SynthSplineParameters):
        shape: List[int] = (128, 128, 128)
        """Shape of the synthetic volume in voxels (x, y, z)."""
        voxel_size: float = 0.02
        """Size of each vx in mm"""
        nb_levels: AnyVar = RandInt(a=1, b=4)
        """Number of hierarchical levels in the vascular tree."""
        tree_density: AnyVar = Uniform(a=0.1, b=0.2)
        """Density of the vascular tree structures per cubic mm^3 (n/mm^3)."""
        tortuosity: AnyVar = Uniform(a=1, b=5)
        """Tortuosity ~= cord / length"""
        radius: AnyVar = Uniform(a=0.1, b=0.15)
        """Mean radius of vascular tree trunk in mm."""
        radius_change: AnyVar = Uniform(a=0.9, b=1.1)
        """Multiplicative variation in radius along the len of the vessel."""
        nb_children: AnyVar = RandInt(a=1, b=4)
        """Number of branches per vessel"""
        radius_ratio: AnyVar = Uniform(a=0.25, b=1)
        """Ratio of the radius of child vessels to parent vessels"""
        device: Union[torch.device, str] = 'cuda'
        """Device to perform computations on, either 'cuda' or 'cpu'."""


class VesselSynthEngineWrapper(object):
    """
    Synthesize 3D vascular labels and save as NIfTI.
    """
    def __init__(self,
                 experiment_dir: str = 'synthetic_data',
                 experiment_number: int = 1,
                 synth_engine: SynthSplineBlock = VesselSynthEngineOCT(),
                 n_volumes: int = 1000,
                 ):
        """
        Initialize the synthesizer.

        Parameters
        ----------
        experiment_dir : str, optional
            Directory for output of synthesis experiments, default is
            'synthetic_data' (which is found in OCT_VESSELSEG_BASE_DIR).
        experiment_number : int, optional
            Identifier for the synthesis experiment, default is 1. (exp0001)
        synth_engine : SynthSplineBlock, optional
            Engine for synthesizing vascular labels, default is
            VesselSynthEngineOCT().
        n_volumes : int, optional
            Number of volumes to synthesize, default is 1000.
        """
        self.synth_engine = synth_engine
        # Set environment variable for JIT (don't move out of this class)
        os.environ['PYTORCH_JIT_USE_NNC_NOT_NVFUSER'] = '1'
        self.shape = synth_engine.shape
        self.n_volumes = n_volumes
        # Output path for the synthesis experiment
        self.experiment_path = (
            f"{vesselseg_outdir}/{experiment_dir}/"
            f"exp{experiment_number:04d}")
        # Create directory for output of synthesis experiment
        # (empty it if it exists)
        PathTools(self.experiment_path).makeDir()
        self.header = nib.Nifti1Header()
        # Serialize and save synthesis parameters
        self.save_params(f'{self.experiment_path}/#_vesselsynth_params.json')
        # Logging init
        logger.info(f"Initialized VesselSynth with {n_volumes} volumes")

    def synth(self):
        """Synthesize a volume of vascular labels and save."""
        # Create a notes file (please write a description of your experiment!)
        logger.info(
            "MAKE SURE YOU EDIT YOUR EXPERIMENT NOTE @ "
            f"{self.experiment_path}/#_notes.txt"
            )
        open(f'{self.experiment_path}/#_notes.txt', 'x').close()
        # Names of synthesized volumes
        synth_names = ['prob', 'label', "level", "nb_levels",
                       "branch", "skeleton"]
        # Loop over number of volumes
        for n in range(self.n_volumes):
            # Log progesss
            print('\n')
            logger.info(f"Creating volume {n:04d}")
            synth_vols = self.synth_engine()
            # Save each volume individually
            for i in range(len(synth_names)):
                synth_vols[i]
                self.save_volume(n, synth_names[i], synth_vols[i])
            # Log progress
            logger.info(f"volume {n:04d} creation complete")

    def save_volume(self, volume_n: int, volume_name: str,
                    volume: torch.Tensor):
        """
        Save the synthesized volume as a NIfTI file.

        Parameters
        ----------
        volume_n : int
            Identifier for the volume.
        volume_name : str
            Name of the synthesized volume type.
        volume : torch.Tensor
            Synthesized volume tensor.
        """
        # Get affine matrix
        affine = default_affine(volume.shape[-3:])
        # Save volume as NIfTI
        nib.save(nib.Nifti1Image(
            volume.squeeze().cpu().numpy(), affine, self.header),
            (f'{self.experiment_path}/'
             f'{volume_n:04d}_vessels_{volume_name}.nii.gz'))
        # TODO: Add logger

    def save_params(self, abspath: str):
        """
        Save the synthesis parameters to a JSON file.

        Parameters
        ---------
        abspath: str
            JSON abspath to log parameters
        """
        # Serialize synthesis parameters
        serialized = self.synth_engine.__dict__['params'].serialize()
        # Convert to JSON string
        json_str = json.dumps(serialized, indent=4)
        # Write to file
        with open(abspath, 'w') as file:
            file.write(json_str)
        # Log parameter saving
        logger.info(
            f"Synthesis parameters saved to {self.experiment_path}/"
            "#_vesselsynth_params.json"
            )


class ImageSynthEngineOCT(nn.Module):
    """
    A module to synthesize OCT-like volumetric images from vessel labels.
    """
    def __init__(self,
                 synth_params: Dict = None,
                 dtype: torch.dtype = torch.float32,
                 device: str = 'cuda',
                 ):
        super().__init__()
        """
        Initialize the ImageSynthEngineOCT for OCT-like volumetric image
        synthesis with parameters optimized for GPU execution.

        Parameters
        ----------
        synth_params : dict, optional
            Arguments for defining the complexity of image synthesis.
        dtype : torch.dtype, optional
            Data type for internal computations.
        device : str, optional
            Device for internal computations, default is 'cuda'.
        """
        self.synth_params = synth_params
        self.dtype = dtype
        self.device = device  # torch.device(
        # device if torch.cuda.is_available() else 'cpu')
        self.backend = dict(device=self.device, dtype=self.dtype)
        self.setup_parameters()

    def setup_parameters(self):
        """
        Setup and initialize synthesis parameters
        """
        self.random_vessel_ablation = self.synth_params['random_vessel_ablation']
        self.speckle_a = float(self.synth_params.get('speckle')[0])
        self.speckle_b = float(self.synth_params['speckle'][1])
        self.gamma_a = float(self.synth_params['gamma'][0])
        self.gamma_b = float(self.synth_params['gamma'][1])
        self.thickness_ = int(self.synth_params['z_decay'][0])
        self.nb_classes_ = int(self.synth_params['parenchyma']['nb_classes'])
        self.shape_ = int(self.synth_params['parenchyma']['shape'])
        self.i_max = float(self.synth_params['imax'])
        self.i_min = float(self.synth_params['imin'])

    # @autocast()  # Enables mixed precision for faster computation
    def forward(self, vessel_labels_tensor: torch.Tensor) -> tuple:
        """
        Generate OCT-like volumetric images.

        Parameters
        ----------
        vessel_labels_tensor : torch.Tensor
            Tensor containing vessel labels with unique integer IDs.
        """
        # Move the tensor to specified device
        vessel_labels_tensor = vessel_labels_tensor.to(self.device)
        # Get sorted list of all unique vessel labels
        vessel_labels = torch.unique(vessel_labels_tensor)
        vessel_labels = vessel_labels[vessel_labels != 0].to(self.device)

        # Randomly make a negative control
        if vessel_labels.numel() >= 2:
            # n_unique_ids = 1
            if RandInt(1, 10)() == 7:
                # vessel_labels_tensor[vessel_labels_tensor > 0] = 0
                vessel_labels_tensor.zero_()
                n_unique_ids = 0
            else:
                if self.random_vessel_ablation:
                    # Hide some vessels randomly
                    n_unique_ids = len(vessel_labels)
                    number_vessels_to_hide = torch.randint(
                        n_unique_ids//10,
                        n_unique_ids-1, [1]
                        ).item()
                    vessel_ids_to_hide = vessel_labels[
                        torch.randperm(n_unique_ids)[:number_vessels_to_hide]]
                    for vessel_id in vessel_ids_to_hide:
                        vessel_labels_tensor[vessel_labels_tensor == vessel_id] = 0
                else:
                    n_unique_ids = len(vessel_labels)
        else:
            vessel_labels_tensor.zero_()
            n_unique_ids = 0

        vessel_mask = torch.clone(vessel_labels_tensor > 0)

        # Synthesize the parenchyma (background tissue)
        parenchyma = self.parenchyma_(vessel_labels_tensor)
        # Optionally, add a DC offset to the parenchyma
        if self.synth_params['dc_offset'] is True:
            dc_offset = Uniform(0, 0.25)()
            parenchyma += dc_offset

        final_volume = parenchyma.clone()
        # Determine if there are any vessels left
        if n_unique_ids > 0:
            # If so, synthesize them (grouped by intensity)
            vessels = self.vessels_(vessel_labels_tensor)
            if self.synth_params['vessel_texture'] is True:
                # Texturize those vessels!!!
                vessel_texture = self.vessel_texture_(vessel_labels_tensor)
                vessels[vessel_mask] *= vessel_texture[vessel_mask]
            # Blending
            alpha = 0.5
            blended_values = torch.clone(final_volume)
            blended_values[vessel_mask] = alpha * vessels[vessel_mask] + (1 - alpha) * final_volume[vessel_mask]

            # Ensure vessels are always darker
            final_volume[vessel_mask] = torch.min(blended_values[vessel_mask], final_volume[vessel_mask] * vessels[vessel_mask])

            # final_volume[vessel_mask] *= vessels[vessel_mask]

        # Normalizing
        final_volume = QuantileTransform()(final_volume)
        # Convert to same dtype as model weights
        final_volume = final_volume.to(torch.float32)
        # Convert one-hot encoded vessels to binary label map
        vessel_labels_tensor[vessel_labels_tensor != 0] = 1

        return final_volume, vessel_labels_tensor

    def parenchyma_(self, vessel_labels_tensor: torch.Tensor) -> torch.Tensor:
        """
        Generate parenchyma (background tissue) based on vessel labels using
        GPU optimized operations.

        Parameters
        ----------
        vessel_labels_tensor : torch.Tensor
            Tensor of vessels with unique ID integer labels.

        Returns
        -------
        torch.Tensor
            Tensor containing the generated parenchyma.
        """
        # Create label map containing unique parenchymal regions with
        # unique IDs.
        parenchyma = RandomSmoothLabelMap(
            nb_classes=RandInt(2, self.nb_classes_)(),
            shape=RandInt(2, self.shape_)(),
            # Add 1 to work w/ every pixel (no 0s)
            )(vessel_labels_tensor).to(self.device) + 1
        # Assign random intensities to parenchyma centered around i
        parenchyma = parenchyma.to(torch.float32)
        for i in torch.unique(parenchyma):
            parenchyma.masked_fill_(parenchyma == i, Normal(i, 0.2)())
        parenchyma /= parenchyma.max()  # Normalize the parenchyma
        # Apply speckle noise
        parenchyma = RandomGammaNoiseTransform(
            sigma=Uniform(self.speckle_a, self.speckle_b)()
            )(parenchyma)

        # Optionally add spherical structures
        if self.synth_params['spheres'] is True:
            # Add first layer of spheres
            if RandInt(0, 2)() == 1:
                spheres = BernoulliDiskTransform(
                    prob=1e-2,
                    radius=RandInt(1, 4)(),
                    value=Uniform(0, 2)()
                    )(parenchyma)[0]
                # Add deformation to spheres
                if RandInt(0, 2)() == 1:
                    spheres = ElasticTransform(shape=5)(spheres).detach()
                parenchyma *= spheres
        # Optionaly apply slabwise banding artifact
        if self.synth_params['slabwise_banding'] is True:
            parenchyma = RandomSlicewiseMulFieldTransform(
                thickness=self.thickness_
                )(parenchyma)
        # Give bias field in lieu of slicewise transform
        elif self.synth_params['slabwise_banding'] is False:
            parenchyma = RandomMulFieldTransform(5)(parenchyma)
        # Apply gamma transformation
        parenchyma = RandomGammaTransform((
            self.gamma_a, self.gamma_b))(parenchyma)
        # Normalize
        parenchyma = QuantileTransform()(parenchyma)
        return parenchyma

    def vessels_(self, vessel_labels_tensor: torch.Tensor) -> torch.Tensor:
        """
        Generate vessel intensities.

        Parameters
        ----------
        vessel_labels_tensor : tensor[int]
            Tensor of vessels with unique integer label IDs.
        """
        # Create a scaling tensor for vessel intensities
        scaling_tensor = torch.zeros(
            vessel_labels_tensor.shape,
            dtype=self.dtype,
            device=vessel_labels_tensor.device)
        # Random factor for vessel texture intensity scaling
        vessel_texture_fix_factor = Uniform(0.5, 1)()
        # Determine how many vessel labels to deal with (after random removal
        # from a previous step)
        vessel_labels_left = torch.unique(vessel_labels_tensor)
        # Iterate through unique vessel labels and assign intensities
        for int_n in vessel_labels_left:
            intensity = Uniform(self.i_min, self.i_max)()
            scaling_tensor.masked_fill_(vessel_labels_tensor == int_n,
                                        intensity * vessel_texture_fix_factor)
        return scaling_tensor

    def vessel_texture_(self,
                        vessel_labels_tensor: torch.Tensor
                        ) -> torch.Tensor:
        """
        Generate intra-vessel textures.

        Parameters
        ----------
        vessel_labels_tensor : tensor[int]
            Tensor containing vessel labels with unique integer IDs.

        Returns
        -------
        torch.Tensor
            Tensor containing the vessel textures.
        """
        # Create unique label map with two unique IDs (regions)
        vessel_texture = RandomSmoothLabelMap(
            nb_classes=Fixed(2),
            shape=self.shape_,
            )(vessel_labels_tensor) + 1  # Add 1 to work w/ every pixel (no 0s)
        # Apply gaussian mixture transformation
        vessel_texture = RandomGaussianMixtureTransform(
            mu=Uniform(0.7, 1)(),
            sigma=0.8,
            dtype=self.dtype
            )(vessel_texture)
        # Normalizing and clamping min
        vessel_texture -= vessel_texture.min()
        vessel_texture /= (vessel_texture.max()*2)
        vessel_texture += 0.5
        vessel_texture.clamp_min_(0)
        return vessel_texture


class ImageSynthEngineWrapper(Dataset):
    """
    Dataset class for synthesizing OCT-like volumetric images from
    vessel labels.
    """
    def __init__(self,
                 exp_path: str = None,
                 synth_params: dict = None,
                 label_type: str = 'label',
                 device: str = "cuda",
                 save_nifti: bool = False,
                 make_fig: bool = False,
                 save_fig: bool = False
                 ):
        """
        Initialize the dataset for synthesizing volumetric OCT images.

        Parameters
        ----------
        exp_path : str
            Path to the experiment directory.
        synth_params : dict
            Parameters for synthesis image synthesis.
        label_type : str
            Type of label to use for synthesis.
        device : str
            Computation device to use.
        save_nifti : bool, optional
            Whether to generate and save volume as a NIfTI file.
        make_fig : bool, optional
            Whether to generate a figure of the synthesized volume.
        save_fig : bool, optional
            Whether to save the generated figure.
        """
        self.exp_path = exp_path
        self.synth_params = synth_params
        self.label_type = label_type
        self.device = device
        self.save_nifti = save_nifti
        self.make_fig = make_fig
        self.save_fig = save_fig

        print(self.exp_path)

        # Set backend
        self.backend = dict(dtype=torch.float32, device=device)
        # Get sorted list of vessel label paths
        self.label_paths = sorted(glob.glob(f"{exp_path}/*label*"))
        # Get sorted list of paths for the ground truth
        if self.label_type == 'label':
            self.y_paths = self.label_paths
        else:
            self.y_paths = sorted(
                glob.glob(f"{self.exp_path}/*{self.label_type}*"))

        # Set paths for saving figures and NIfTI files
        self.sample_fig_dir = f"{exp_path}/sample_vols/figures"
        self.sample_nifti_dir = f"{exp_path}/sample_vols/niftis"
        # Create directories for saving figures and NIfTI files
        PathTools(self.sample_nifti_dir).makeDir()
        PathTools(self.sample_fig_dir).makeDir()

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.
        """
        return len(self.label_paths)

    def __getitem__(self, idx: int) -> tuple:
        """
        Retrieve a synthesized OCT volume and label.

        Parameters
        ----------
        idx : int
            Index of the sample from disk.
        """
        # Load the vessel label NIfTI file and its affine transformation
        label_nifti = nib.load(self.label_paths[idx])
        label_affine = label_nifti.affine

        # Convert the NIfTI data to a tensor
        self.label_tensor_backend = dict(device='cuda', dtype=torch.int32)
        label_tensor = torch.from_numpy(label_nifti.get_fdata()).to(
            **self.label_tensor_backend)
        # Clip the tensor values to a valid range and add a new dimension
        label_tensor = torch.clip(label_tensor, 0, 32767)[None]

        # Make instance of ImageSynthEngine and generate the synthesized volume
        im, prob = ImageSynthEngineOCT(
            synth_params=self.synth_params
            )(label_tensor)
        # Convert the tensor data to numpy arrays
        im = im.detach().cpu().numpy().squeeze()
        prob = prob.to(torch.int32).cpu().numpy().squeeze()

        # Optionally save the synthesized volume and probability map as NIfTIs
        if self.save_nifti is True:
            volume_name = f"volume-{idx:04d}"
            out_path_volume = f'{self.sample_nifti_dir}/{volume_name}.nii'
            out_path_prob = f'{self.sample_nifti_dir}/{volume_name}_MASK.nii'
            logger.info(f"Saved NIfTI to: {out_path_volume}")
            nib.save(nib.Nifti1Image(im, affine=label_affine), out_path_volume)
            nib.save(nib.Nifti1Image(prob, affine=label_affine), out_path_prob)

        # Optionally generate and save figures
        if self.save_fig is True:
            self.make_fig = True
        if self.save_fig is True:
            self._make_fig(im, prob)
        if self.save_nifti is True:
            plt.savefig(f"{self.sample_fig_dir}/{volume_name}.png")

        # return im.unsqueeze(0), prob.unsqueeze(0)

    def _make_fig(self, im: np.ndarray, prob: np.ndarray) -> None:
        """
        Make 2D figure (GT, prediction, gt-pred superimposed) and display it.

        Parameters
        ----------
        im : arr[float]
            Volume of x data
        prob: arr[bool]
            Volume of y data
        """
        plt.figure()
        # Create subplots for image, probability map, and superimposed image
        f, axarr = plt.subplots(nrows=1, ncols=3, figsize=(15, 15),
                                constrained_layout=True)
        axarr = axarr.flatten()
        frame = np.random.randint(0, im.shape[0])
        # Display the original image
        axarr[0].imshow(im[frame], cmap='gray')
        # Display the probability map
        axarr[1].imshow(prob[frame], cmap='gray')
        # Display the superimposed image and probability map
        axarr[2].imshow(im[frame], cmap='gray')
        axarr[2].contour(prob[frame], cmap='magma', alpha=1)


class VesselLabelDataset(Dataset):
    """
    Dataset for loading and processing 3D vascular networks.
    """
    def __init__(self,
                 inputs,
                 subset=-1,
                 ):
        """
        Initialize the dataset with the given inputs and subset size.

        Parameters
        ----------
        inputs : list or str
            List of file paths or a directory/pattern representing the input
            labels.
        subset : int, optional
            Number of examples to consider for the dataset, if not all.

        Returns
        -------
        label as shape as torch.Tensor of shape (1, n, n, n)
        """
        self.subset = slice(subset)
        # Uses less RAM in multithreads
        if isinstance(inputs, str):
            inputs = glob.glob(inputs)
        elif not isinstance(inputs, (list, np.ndarray)):
            raise TypeError("inputs must be a list, numpy array, or a glob\
                pattern string")
        self.inputs = np.asarray(inputs[self.subset])

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        """
        Get a single label from the dataset.

        Parameters
        ----------
        idx : int
            Index of the label to retrieve.
        """
        label = torch.from_numpy(
            nib.load(self.inputs[idx]).get_fdata()).to('cuda')
        label = torch.clip(label, 0, 32767).to(torch.int16)[None]
        return label

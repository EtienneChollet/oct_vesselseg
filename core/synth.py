__all__ = [
    'VesselSynthEngineOCT',
    'VesselSynthEngineWrapper',
]

# Standard imports
import os
import json
import torch
import logging
from typing import Union, List

# Balbasty imports
import nibabel as nib
from synthspline.random import Uniform, RandInt, AnyVar
from synthspline.utils import default_affine
from synthspline.labelsynth import SynthSplineParameters, SynthSplineBlock

# Custom Imports
from core.utils import PathTools

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] (%(levelname)s) ==> %(message)s',
    datefmt='%Y-%m-%d | %I:%M:%S %p'
)
# Init logger
logger = logging.getLogger(__name__)


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
    Synthesize 3D vascular network and save as NIfTI.
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
            'synthetic_data' (which is found in oct_vesselseeg/output).
        experiment_number : int, optional
            Identifier for the synthesis experiment, default is 1. (exp0001)
        synth_engine : SynthSplineBlock, optional
            Engine for synthesizing vascular network, default is
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
        self.experiment_path = (f"output/{experiment_dir}/"
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
        """Synthesize a vascular network and save each volume."""
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
        # Log saved volume
        # logger.info(
        #    f"Saved volume {volume_n:04d}_vessels_{volume_name}.nii.gz")

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

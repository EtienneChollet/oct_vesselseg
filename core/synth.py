__all__ = [
    'VesselSynth',
]

# Standard imports
import os
import json
import torch


# Custom Imports
from utils import PathTools

class VesselSynth(object):
    """
    Synthesize 3D vascular network and save as nifti.
    """
    def __init__(self,
                 device: str = 'cuda',
                 json_param_path: str = ('scripts/1_vesselsynth/'
                                         'vesselsynth_params.json'),
                 experiment_dir: str = 'synthetic_data',
                 experiment_number: int = 1
                 ):
        """
        Initialize the VesselSynth class to synthesize 3D vascular networks.

        Parameters
        ----------
        device : str, optional
            Which device to run computations on, default is 'cuda'.
        json_param_path : str, optional
            Path to JSON file containing parameters.
        experiment_dir : str, optional
            Directory for output of synthetic experiments.
        experiment_number : int, optional
            Identifier for the experiment.
        """
        # All JIT things need to be handled here. Do not put them outside
        # this class.
        os.environ['PYTORCH_JIT_USE_NNC_NOT_NVFUSER'] = '1'
        backend.jitfields = True
        self.device = device
        self.json_params = json.load(open(json_param_path))
        self.shape = self.json_params['shape']                           
        self.n_volumes = self.json_params['n_volumes']
        self.begin_at_volume_n = self.json_params['begin_at_volume_n']
        self.experiment_path = (f"output/{experiment_dir}/"
                                f"exp{experiment_number:04d}")
        PathTools(self.experiment_path).makeDir()
        self.header = nib.Nifti1Header()
        self.prepOutput(f'{self.experiment_path}/#_vesselsynth_params.json')
        self.backend()
        self.outputShape()

    def synth(self):
        """
        Synthesize a vascular network.
        """
        file = open(f'{self.experiment_path}/#_notes.txt', 'x')
        file.close()

        for n in range(self.begin_at_volume_n, self.begin_at_volume_n
                       + self.n_volumes):
            print(f"Making volume {n:04d}")
            synth_names = ['prob',
                           'label',
                           "level",
                           "nb_levels",
                           "branch",
                           "skeleton"]
            # Synthesize volumes
            synth_vols = SynthVesselOCT(shape=self.shape, device=self.device)()
            # Save each volume individually
            for i in range(len(synth_names)):
                self.saveVolume(n, synth_names[i], synth_vols[i])

    def backend(self):
        """
        Check and set the computation device.
        """
        self.device = torch.device(self.device)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            print('CUDA not available, using CPU.')
            self.device = 'cpu'

    def outputShape(self):
        """
        Ensure shape is a list of three dimensions.
        """
        if not isinstance(self.shape, list):
            self.shape = [self.shape]
        while len(self.shape) < 3:
            self.shape += self.shape[-1:]

    def saveVolume(self, volume_n: int, volume_name: str,
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
        affine = default_affine(volume.shape[-3:])
        nib.save(nib.Nifti1Image(
            volume.squeeze().cpu().numpy(), affine, self.header),
            (f'{self.experiment_path}/'
             f'{volume_n:04d}_vessels_{volume_name}.nii.gz'))

    def prepOutput(self, abspath: str):
        """
        Clear files in output dir and log synth parameters to json file.

        Parameters
        ---------
        abspath: str
            JSON abspath to log parameters
        """
        json_object = json.dumps(self.json_params, indent=4)
        with open(abspath, 'w') as file:
            file.write(json_object)

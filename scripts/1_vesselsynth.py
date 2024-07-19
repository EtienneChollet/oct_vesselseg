import torch
from core.synth import VesselSynth, SynthVesselOCT
from synthspline.random import Uniform, RandInt

synth_params = {
    'shape': (128, 128, 128),
    'voxel_size': 0.02,
    'nb_levels': RandInt(1, 4),
    'tree_density': Uniform(0.1, 0.2),
    'tortuosity': Uniform(1, 5),
    'radius': Uniform(0.1, 0.15),
    'radius_change': Uniform(0.9, 1.1),
    'nb_children': RandInt(1, 4),
    'radius_ratio': Uniform(0.25, 1),
    'device': 'cuda'
}

# TODO: Make cli parser?
if __name__ == "__main__":
    torch.no_grad()
    synth_block = SynthVesselOCT(**synth_params)
    VesselSynth(
        experiment_number=1,
        synth_block=synth_block,
        ).synth()

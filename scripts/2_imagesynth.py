# Custom Imports
from core.synth import ImageSynthEngineWrapper
import time

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
    synth = ImageSynthEngineWrapper(
        exp_path="output/synthetic_data/exp0001",
        synth_params=synth_params,
        save_nifti=True,
        save_fig=True
        )
    t1 = time.time()
    for i in range(10):
        synth[i]
    t2 = time.time()
    print(t2-t1)

"""
Client for the FastAPI 3-D segmentation.

Example
-------
python oct_vesselseg/client.py /autofs/cluster/octdata2/users/epc28/data/caroline_data/I46_Somatosensory_20um_averaging_new_mask_cleaned.nii
"""

from typing import Literal
import requests
from cyclopts import App

app = App()

@app.default()
def predict_volume(
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


if __name__ == "__main__":
    app()

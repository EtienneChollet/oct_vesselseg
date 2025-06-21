"""
Client for the FastAPI 3-D segmentation.

Example
-------
python oct_vesselseg/client.py /autofs/cluster/octdata2/users/epc28/data/caroline_data/I46_Somatosensory_20um_averaging_new_mask_cleaned.nii
"""

import requests
from cyclopts import App

app = App()


@app.default()
def predict_volume(
    in_path: str,
    out_path: str | None = 'oct_vesselseg_output.nii.gz',
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
    api_url : str, optional
        Full URL to the `/predict` endpoint, e.g.
        "http://127.0.0.1:8000/predict".
    """

    data = {
        "in_path": in_path,
        "out_path": out_path
    }

    with requests.post(api_url, data=data) as resp:
        resp.raise_for_status()
        if out_path:
            print(f"Saved prediction to {out_path!r}")


if __name__ == "__main__":
    app()

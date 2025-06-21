"""
Serve a 3-D oct_vesselseg model over HTTP.

Run
----
>>> pip install fastapi uvicorn python-multipart
>>> cd oct_vesselseg/oct_vesselseg/
>>> uvicorn serve_predict:app --host 0.0.0.0 --port 8000 --reload

Example client
--------------
>>> curl -X POST http://127.0.0.1:8000/predict \
    -F "in_path=/autofs/cluster/octdata2/users/epc28/data/caroline_data/I46_Somatosensory_20um_averaging_new_mask_cleaned.nii" \
    -F "out_path=/autofs/cluster/octdata2/users/epc28/data/caroline_data/I46_Somatosensory_20um_averaging_new_mask_cleaned_prediction.nii"
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated
import nibabel as nib
from fastapi import FastAPI, Form
from fastapi.responses import StreamingResponse


# Initialize app
app = FastAPI(
    title="3-D UNet Segmentation API",
    version="0.1.0",
    description="Point to a NIfTI volume anywhere on the Martino's cluster! Get a vessel segmentation back!.",
)


@app.post("/predict")
async def predict(
    in_path: Annotated[str, Form()],
    out_path: Annotated[str, Form()],
) -> StreamingResponse:
    """
    Segment a 3-D volume.

    Parameters
    ----------
    in_path
        NIfTI file (`.nii` / `.nii.gz`) containing volume of shape (D, H, W) or
        (C, D, H, W).

    Returns
    -------
    StreamingResponse
        Binary payload with appropriate `Content-Type`.
    """

    print(f'Loading volume at: {in_path}')
    volume_tensor = nib.load(in_path).get_fdata()
    print(volume_tensor.shape)
    print(f'Saving to {out_path}')

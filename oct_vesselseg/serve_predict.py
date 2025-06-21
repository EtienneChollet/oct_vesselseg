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
import io
import sys

from enum import Enum
from typing import Annotated
import nibabel as nib
from fastapi import FastAPI, Form, Request
from fastapi.responses import StreamingResponse

import gc
import time
import torch
from oct_vesselseg.models import UnetWrapper
from oct_vesselseg.data import RealOctPredict, RealOctConfig


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
    print(f'Going to save to {out_path}')

    volume_tensor = nib.load(in_path).get_fdata()
    print(volume_tensor.shape)

    with torch.no_grad():
        unet = UnetWrapper(
            version_n=1,
            model_dir='models',
            device='cuda'
        )

        unet.load(type='best', mode='test')

        # Configuring prediction
        oct_config = RealOctConfig(
            input=in_path,
            patch_size=128,
            redundancy=3,
            pad_it=True,
            padding_method='reflect',
            normalize=True,
        )

        prediction = RealOctPredict(oct_config, trainee=unet.trainee)
        prediction.predict_on_all()


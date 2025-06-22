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
import shutil
import threading
from typing import Generator
import asyncio
from typing import Literal

from enum import Enum
from typing import Annotated
import nibabel as nib
from fastapi import FastAPI, Form, Request
from fastapi.responses import StreamingResponse

import gc
import time
import torch
from oct_vesselseg.server import run_job_in_thread, Queue, monitor_disconnect
from oct_vesselseg.models import UnetWrapper
from oct_vesselseg.data import RealOctPredict, RealOctConfig
from oct_vesselseg.callbacks import CancelOnFlag


# Initialize app
app = FastAPI(
    title="3-D UNet Segmentation API",
    version="0.1.0",
    description="Point to a NIfTI volume anywhere on the Martino's cluster! Get a vessel segmentation back!.",
)


@app.post("/predict")
async def predict(
    request: Request,
    in_path: Annotated[str, Form()],
    out_path: Annotated[str, Form()],
    patch_size: Annotated[int, Form()] = 128, 
    redundancy: Annotated[int, Form()] = 3,
    pad_it: Annotated[Literal['true', 'false'], Form()] = 'true',
    padding_method: Annotated[str, Form()] = 'reflect',
    normalize_patches: Annotated[Literal['true', 'false'], Form()] = 'true',
):
    queue = Queue()
    cancel_event = threading.Event()

    # Convert string -> bool
    pad_it = pad_it.lower() == "true"
    normalize_patches = normalize_patches.lower() == "true"

    # Start the background job
    run_job_in_thread(
        fn=prediction_job,
        queue=queue,
        cancel_event=cancel_event,
        in_path=in_path,
        out_path=out_path,
        patch_size=patch_size,
        redundancy=redundancy,
        pad_it=pad_it,
        padding_method=padding_method,
        normalize_patches=normalize_patches,
    )

    # Start a disconnect monitor in parallel
    asyncio.create_task(monitor_disconnect(request, cancel_event))

    async def stream_from_queue() -> Generator[str, None, None]:
        while True:
            msg = await asyncio.to_thread(queue.get)
            if msg is None:
                break
            yield msg + "\n"

    return StreamingResponse(stream_from_queue(), media_type="text/plain")


def prediction_job(
    cancel_event: threading.Event,
    in_path: str,
    out_path: str = None,
    patch_size: int = 128, 
    redundancy: int = 3,
    pad_it: bool = True,
    padding_method: str = 'reflect',
    normalize_patches: bool = True,
    #*args, **kwargs,
):

    print("=== Job Started ===\n")
    print(f"Loading volume: {in_path}\n")
    print(f'Going to save to {out_path}\n')

    volume_tensor = nib.load(in_path).get_fdata()
    print(f"Shape: {volume_tensor.shape}\n")

    callbacks = [
        CancelOnFlag(cancel_event),
    ]

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
            patch_size=patch_size,
            redundancy=redundancy,
            pad_it=pad_it,
            padding_method=padding_method,
            normalize=normalize_patches,
        )

        prediction = RealOctPredict(
            oct_config,
            trainee=unet.trainee,
            callbacks=callbacks,
        )

        prediction.predict_on_all()
        prediction.save_prediction(out_path)

        del unet, oct_config, volume_tensor, prediction
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()



    print("=== Job Completed ===\n")

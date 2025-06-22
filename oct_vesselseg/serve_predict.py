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
):
    queue = Queue()
    cancel_event = threading.Event()

    # Start the background job
    run_job_in_thread(
        fn=prediction_job,
        queue=queue,
        cancel_event=cancel_event,
        in_path=in_path,
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
):

    print("=== Job Started ===")
    print(f"\nLoading volume: {in_path}\n")
    print(f'Going to save to {out_path}\n')

    volume_tensor = nib.load(in_path).get_fdata()
    print(f"Shape: {volume_tensor.shape}\n")

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

    #for i in range(1000):
    #    if cancel_event.is_set():
    #        print(f"Cancelled at step {i}")
    #        return
    #    print(f"Working... step {i}")
    #    time.sleep(0.5)
    print("=== Job Completed ===")


# @app.post("/predict")
async def predict3333333(
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

    async def streamer():

        width = shutil.get_terminal_size(fallback=(80, 24)).columns
        yield str("-" * width).encode()
        yield f"\nLoading volume: {in_path}\n".encode()
        print(f'Going to save to {out_path}')

        volume_tensor = nib.load(in_path).get_fdata()
        yield f"Shape: {volume_tensor.shape}\n".encode()

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
            # prediction.predict_on_all()

        del unet, oct_config, volume_tensor, prediction
        torch.cuda.empty_cache()
        gc.collect()

        print('Finished the job!')

    return StreamingResponse(
        streamer(),
        media_type="application/octet-stream"
    )

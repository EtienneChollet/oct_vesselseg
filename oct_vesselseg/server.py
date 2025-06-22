import asyncio
import sys
import threading
from queue import Queue

from fastapi import Request


class DualWriter:
    """Print to both server logs and client stream."""
    def __init__(self, queue: Queue):
        self.queue = queue
        self._console = sys.__stdout__

    def write(self, msg: str):
        if msg.strip():
            self.queue.put(msg)
            self._console.write(msg)
            self._console.flush()

    def flush(self):
        self._console.flush()


def run_job_in_thread(
    fn,
    queue: Queue,
    cancel_event: threading.Event,
    *args,
    **kwargs,
) -> threading.Thread:

    def wrapper():

        original_stdout = sys.stdout
        sys.stdout = DualWriter(queue)
        try:
            fn(
                cancel_event=cancel_event, *args, **kwargs)
        finally:
            sys.stdout = original_stdout
            queue.put(None)

    thread = threading.Thread(target=wrapper)
    thread.start()
    return thread


async def monitor_disconnect(
    request: Request,
    cancel_event: threading.Event
):
    while not cancel_event.is_set():
        if await request.is_disconnected():
            cancel_event.set()
            print("Client disconnected — cancelling job.")
            break
        await asyncio.sleep(0.1)

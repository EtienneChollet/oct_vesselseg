from abc import ABC, abstractmethod
import time
import sys
import threading

class InferenceCallback(ABC):
    @abstractmethod
    def on_step(self, step_num: int) -> bool:
        pass

class InferenceETA(InferenceCallback):

    def __init__(self, total_num_patches: int):
        self.total_num_patches = total_num_patches

    def on_step(self, step_num: int, t0: float):

        # Print updates every ten patches
        if (step_num + 1) % 10 == 0:
            total_elapsed_time = time.time() - t0
            avg_pred_time = round(total_elapsed_time / (step_num + 1), 3)

            total_pred_time = round(
                avg_pred_time * self.total_num_patches / 60, 2
            )

            # Construct the status message
            status_message = (
                f"\rPrediction {step_num + 1}/{self.total_num_patches} | "
                f"{avg_pred_time} sec/pred | "
                f"{total_pred_time} min total pred time"
                )
            sys.stdout.write(status_message)
            sys.stdout.flush()

class CancelOnFlag:
    def __init__(self, cancel_event: threading.Event):
        self.cancel_event = cancel_event
        self.should_stop = False

    def on_step(self, step_num: int, *args, **kwargs):
        if self.cancel_event.is_set():
            print(f"[CancelOnFlag] Cancelling at step {step_num}\n")
            self.should_stop = True

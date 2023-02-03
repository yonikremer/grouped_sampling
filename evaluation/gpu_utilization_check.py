from threading import Thread, Event
from typing import Callable
from warnings import warn

# noinspection PyUnresolvedReferences
from nvidia_smi import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates


def check_gpu_utilization(func: Callable) -> Callable:
    """
    A decorator that checks the GPU utilization during the execution of the wrapped function
    And prints a warning if it is zero for interval consecutive seconds
    """
    def wrapper(*args, **kwargs):
        # Initialize the NVML library
        nvmlInit()

        # Get the handle to the GPU device
        device_index = 0  # index of the GPU device to use
        handle = nvmlDeviceGetHandleByIndex(device_index)

        # Start the GPU utilization checking thread
        stop_flag = Event()
        utilization_thread = Thread(target=_check_utilization, args=(handle, stop_flag))
        utilization_thread.start()

        # Run the original function
        result = func(*args, **kwargs)

        # Stop the GPU utilization checking thread
        stop_flag.set()
        utilization_thread.join()

        return result

    return wrapper


def _check_utilization(handle: nvmlDeviceGetHandleByIndex, stop_flag: Event) -> None:
    """
    A thread that checks the GPU utilization every second
    during the execution of the wrapped function
    and sends a warning if the gpu utilization is zero for interval seconds
    """
    while not stop_flag.is_set():
        # Get the GPU utilization using nvidia_smi
        utilization = nvmlDeviceGetUtilizationRates(handle)
        gpu_utilization = utilization.gpu
        if gpu_utilization > 0:
            return
    # Print a warning if the GPU utilization is zero
    warn("GPU utilization is zero")

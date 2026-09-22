from __future__ import annotations

import os


def fix_ld_library_path():
    new_path = "/home/yoni/miniconda3/envs/grouped_sampling_new/lib/python3.10/site-packages/nvidia/cuda_runtime/lib"
    if "LD_LIBRARY_PATH" in os.environ:
        os.environ["LD_LIBRARY_PATH"] += ":" + new_path
    else:
        os.environ["LD_LIBRARY_PATH"] = new_path
    import bitsandbytes  # noqa: PLC0415 - must import after LD_LIBRARY_PATH is set
    print("bitsandbytes version:", bitsandbytes.__version__)


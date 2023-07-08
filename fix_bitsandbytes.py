import os


def fix_ld_library_path():
    new_path = "/home/yoni/miniconda3/envs/grouped_sampling_new/lib/python3.10/site-packages/nvidia/cuda_runtime/lib"
    if "LD_LIBRARY_PATH" in os.environ:
        os.environ["LD_LIBRARY_PATH"] += ":" + new_path
    else:
        os.environ["LD_LIBRARY_PATH"] = new_path
    import bitsandbytes
    if not bitsandbytes.COMPILED_WITH_CUDA:
        raise RuntimeError("bitsandbytes was not compiled with CUDA support")

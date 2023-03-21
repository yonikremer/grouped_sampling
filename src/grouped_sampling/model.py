from warnings import warn

from torch import cuda
from torch.cuda import OutOfMemoryError
from transformers import AutoModelForCausalLM, PreTrainedModel

from .llama.modeling_llama import LLaMAForCausalLM


def get_model(
    model_name: str,
    load_in_8bit: bool = False,
    use_cuda: bool = True,
    **kwargs,
) -> PreTrainedModel:
    using_8bit = use_cuda and load_in_8bit and cuda.is_available()
    if using_8bit:
        full_model_kwargs = {
            "pretrained_model_name_or_path": model_name,
            "load_in_8bit": using_8bit,
            "device_map": "auto",
            "resume_download": True,
            **kwargs,
        }
    else:
        full_model_kwargs = {
            "pretrained_model_name_or_path": model_name,
            "load_in_8bit": using_8bit,
            **kwargs,
        }
    try:
        model = AutoModelForCausalLM.from_pretrained(**full_model_kwargs)
    except KeyError as e:
        if "llama" in model_name.lower():
            model = LLaMAForCausalLM.from_pretrained(**full_model_kwargs)
        else:
            raise e
    if use_cuda and cuda.is_available():
        try:
            model = model.cuda()
        except OutOfMemoryError:
            warn("The model is too large for your GPU, using the CPU instead.")
    else:
        warn("CUDA is not avilable, using the CPU instead")
    return model

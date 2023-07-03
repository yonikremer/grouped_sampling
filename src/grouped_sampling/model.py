from warnings import warn

from torch import cuda, compile
from torch._dynamo import OptimizedModule
from torch.cuda import OutOfMemoryError
from transformers import AutoModelForCausalLM


def get_model(
        model_name: str,
        load_in_8bit: bool = False,
        **kwargs,
) -> OptimizedModule:
    using_8bit = load_in_8bit and cuda.is_available()
    full_model_kwargs = {
        "pretrained_model_name_or_path": model_name,
        "load_in_8bit": using_8bit,
        "resume_download": True,
        **kwargs,
    }
    if using_8bit:
        full_model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(**full_model_kwargs)
    if cuda.is_available():
        try:
            model = model.cuda()
        except OutOfMemoryError:
            warn("The model is too large for your GPU, using the CPU instead.")
    else:
        warn("CUDA is not avilable, using the CPU instead")
    model = model.eval()
    model = compile(model)
    return model

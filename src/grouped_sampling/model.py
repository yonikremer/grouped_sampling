from warnings import warn

from huggingface_hub.utils import RepositoryNotFoundError
from torch import compile, cuda, inference_mode
from torch._dynamo import OptimizedModule
from torch.cuda import OutOfMemoryError
from transformers import AutoModelForCausalLM


@inference_mode()
def get_model(
    model_name: str,
    load_in_8bit: bool = False,
    **kwargs,
) -> OptimizedModule:
    """
    Load a model from the huggingface model hub, and compile it for faster inference.
    args:
        model_name: the name of the model to load.
        load_in_8bit: if True, the model will be loaded in 8bit mode, which is faster but less accurate.
        **kwargs: additional arguments to pass to the model's from_pretrained method.
    returns:
        the loaded model OptimizedModule.
    raises: huggingface_hub.utils.RepositoryNotFoundError if the model is not found.
    """
    using_8bit = load_in_8bit and cuda.is_available()
    full_model_kwargs = {
        "pretrained_model_name_or_path": model_name,
        "load_in_8bit": using_8bit,
        "resume_download": True,
        **kwargs,
    }
    if using_8bit:
        full_model_kwargs["device_map"] = "auto"
    try:
        model = AutoModelForCausalLM.from_pretrained(**full_model_kwargs)
    # If the model doesn't support exists
    except OSError as error:
        raise RepositoryNotFoundError(
            f"Model {model_name} not found in the model hub.\n"
            "If you are trying to use a local model, make sure to use the full path.\n"
            "If you are trying to load a private model, make sure to pass your huggingface token."
            + str(error))
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

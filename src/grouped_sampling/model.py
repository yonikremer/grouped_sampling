from huggingface_hub.utils import RepositoryNotFoundError
from torch import inference_mode

from torch.nn import Module
from transformers import AutoModelForCausalLM


@inference_mode()
def get_model(
    model_name: str,
    **kwargs,
) -> Module:
    """
    Load a model from the huggingface model hub, and compile it for faster inference.
    args:
        model_name: the name of the model to load.
        **kwargs: additional arguments to pass to the model's from_pretrained method.
    returns:
        the loaded model OptimizedModule.
    raises:
        huggingface_hub.utils.RepositoryNotFoundError if the model is not found.
        torch.cuda.OutOfMemoryError if the model uses too much memory.
    """
    full_model_kwargs = {
        "pretrained_model_name_or_path": model_name,
        "resume_download": True,
        "device_map": "auto",
        **kwargs,
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(**full_model_kwargs)
    # If the model doesn't support exists
    except OSError as error:
        raise RepositoryNotFoundError(
            f"Model {model_name} not found in the model hub.\n"
            "If you are trying to use a local model, make sure to use the full path.\n"
            "If you are trying to load a private model, make sure to pass your huggingface token."
            + str(error),
            response=None,
        )
    model = model.eval()
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    return model

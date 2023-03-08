from warnings import warn

from torch import cuda
from torch.cuda import OutOfMemoryError
from transformers import PreTrainedModel, AutoModelForCausalLM

from .llama.modeling_llama import LLaMAModel


def get_model(
        model_name: str,
        load_in_8bit: bool = True,
        **kwargs,
) -> PreTrainedModel:
    print("load in 8bit: " + str(load_in_8bit))
    try:
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_name,
            load_in_8bit=load_in_8bit and cuda.is_available(),
            device_map="auto",
            offload_folder="offload",
            offload_state_dict=True,
            **kwargs,
        )
    except KeyError as e:
        if "llama" in model_name.lower():
            model = LLaMAModel.from_pretrained(
                pretrained_model_name_or_path=model_name,
                load_in_8bit=load_in_8bit and cuda.is_available(),
                device_map="auto",
                offload_folder="offload",
                offload_state_dict=True,
                **kwargs,
            )
        else:
            raise e
    if cuda.is_available():
        try:
            model = model.cuda()
        except OutOfMemoryError:
            warn("The model is too large for your GPU, using the CPU instead.")
    else:
        warn("CUDA is not avilable, using the CPU instead")
    return model

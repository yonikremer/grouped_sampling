from torch import cuda
from transformers import PreTrainedModel, AutoModelForCausalLM

from .llama.modeling_llama import LLaMAModel


def get_model(
        model_name: str,
        load_in_8bit: bool = True,
        **kwargs,
) -> PreTrainedModel:
    print("load in 8bit: " + str(load_in_8bit))
    try:
        return AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_name,
            load_in_8bit=load_in_8bit and cuda.is_available(),
            device_map="auto",
            offload_folder="offload",
            offload_state_dict=True,
            **kwargs,
        )
    except KeyError as e:
        if "llama" in model_name.lower():
            return LLaMAModel.from_pretrained(
                pretrained_model_name_or_path=model_name,
                load_in_8bit=load_in_8bit and cuda.is_available(),
                device_map="auto",
                offload_folder="offload",
                offload_state_dict=True,
                **kwargs,
            )
        raise e

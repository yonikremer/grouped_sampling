import multiprocessing
import os
from functools import lru_cache
from typing import Iterable, List, Optional, Set

from cache_to_disk import cache_to_disk, delete_disk_caches_for_function
from huggingface_hub import ModelFilter, hf_api
from huggingface_hub.hf_api import ModelInfo

SUPPORTED_MODEL_NAME_PAGES_FORMAT = (
    "https://huggingface.co/models?pipeline_tag=text-generation&library=pytorch"
)
DEFAULT_MIN_NUMBER_OF_DOWNLOADS = 0
DEFAULT_MIN_NUMBER_OF_LIKES = 0


@lru_cache(maxsize=1)
def get_unsupported_model_names() -> Set[str]:
    """returns the set of unsupported model names"""
    this_file_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(this_file_dir, "unsupported_models.txt"), "r") as f:
        return set(f.read().splitlines())


@lru_cache(maxsize=1)
def get_unsupported_organization_names() -> Set[str]:
    """returns the set of unsupported organization names"""
    this_file_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(this_file_dir, "unsupported_organizations.txt"),
              "r") as f:
        return set(f.read().splitlines())


def model_info_filter(model_info: ModelInfo, ) -> Optional[str]:
    """returns True if the model is supported, False otherwise"""
    model_id = model_info.modelId
    if model_id in get_unsupported_model_names():
        return None
    if model_id.split("/")[0] in get_unsupported_organization_names():
        return None
    if "mt0" in model_id or "santacoder" in model_id or "8bit" in model_id:
        return None
    return model_id


@cache_to_disk(n_days_to_cache=30)
def get_full_models_list() -> List[str]:
    models: Iterable[ModelInfo] = hf_api.list_models(filter=ModelFilter(
        task="text-generation",
        library="pytorch",
    ), )
    with multiprocessing.Pool() as pool:
        supported_model_names = pool.map(
            iterable=models,
            func=model_info_filter,
        )
    return [
        model_name for model_name in supported_model_names
        if model_name is not None
    ]


def unsupported_model_name_error_message(model_name: str) -> str:
    return f"""{model_name} is not supported yet.
        The algorithm only supports models that appear here: {SUPPORTED_MODEL_NAME_PAGES_FORMAT}.
        with at least {DEFAULT_MIN_NUMBER_OF_DOWNLOADS} downloads and {DEFAULT_MIN_NUMBER_OF_LIKES} likes.
        That wasn't created by one of the following organizations: {get_unsupported_organization_names()}.
        And that aren't one of the following models: {get_unsupported_model_names()}.
        And that have a non empty model card without a waring.
        And they are not a santacoder mt0 or an 8bit model."""


@cache_to_disk(n_days_to_cache=30)
def is_supported(model_name: str, ) -> bool:
    return model_name in get_full_models_list()


class UnsupportedModelNameException(Exception):

    def __init__(self, model_name: str):
        super().__init__(unsupported_model_name_error_message(model_name))


def check_support(model_name: str) -> None:
    if not is_supported(model_name):
        raise UnsupportedModelNameException(model_name)


if __name__ == "__main__":
    delete_disk_caches_for_function("get_full_models_list")
    delete_disk_caches_for_function("is_supported")

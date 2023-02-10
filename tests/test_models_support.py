import os
from typing import Generator, Set, Union, List

import pytest
import requests
from bs4 import BeautifulSoup, Tag, NavigableString, PageElement
from concurrent.futures import ThreadPoolExecutor, as_completed

from transformers import AutoTokenizer, AutoConfig

from src.grouped_sampling import GroupedSamplingPipeLine
from src.grouped_sampling.base_pipeline import get_padding_id, get_end_of_text_id

SUPPORTED_MODEL_NAME_PAGES_FORMAT = "https://huggingface.co/models?pipeline_tag=text-generation&library=pytorch"
MAX_WORKERS = 10
MIN_NUMBER_OF_DOWNLOADS = 100
MIN_NUMBER_OF_LIKES = 20
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def is_blacklisted_model(model_name: str) -> bool:
    blacklisted_model_names = {
        "ykilcher/gpt-4chan",
        "bigcode/santacoder",
        "hivemind/gpt-j-6B-8bit",
        "openai-gpt",
    }
    return model_name in blacklisted_model_names


def requires_additional_library(model_name: str) -> bool:
    models_that_requires_libraries = {
        "microsoft/biogpt",
        "microsoft/BioGPT-Large",
        "microsoft/BioGPT-Large-PubMedQA",
        "rinna/japanese-gpt-1b",
        "rinna/japanese-gpt2-medium"
    }
    return model_name in models_that_requires_libraries


def get_model_name(model_card: Tag) -> str:
    h4_class = "text-md truncate font-mono text-black dark:group-hover:text-yellow-500 group-hover:text-indigo-600"
    h4_tag = model_card.find("h4", class_=h4_class)
    return h4_tag.text


def is_a_number(s: PageElement) -> bool:
    s = s.text.strip().lower().replace("k", "").replace("m", "").replace(",", "").replace(".", "").replace("b", "")
    try:
        float(s)
        return True
    except ValueError:
        return False


def get_numeric_contents(model_card):
    div: Union[Tag | NavigableString] = model_card.find(
        "div",
        class_="mr-1 flex items-center overflow-hidden whitespace-nowrap text-sm leading-tight text-gray-400",
        recursive=True
    )
    contents: List[PageElement] = div.contents
    contents_without_tags: List[PageElement] = [content for content in contents if not isinstance(content, Tag)]
    number_contents: List[PageElement] = [content for content in contents_without_tags if is_a_number(content)]
    return number_contents


def convert_to_int(element: PageElement) -> int:
    element_str = element.text.strip().lower()
    if element_str.endswith("k"):
        return int(float(element_str[:-1]) * 1_000)
    elif element_str.endswith("m"):
        return int(float(element_str[:-1]) * 1_000_000)
    elif element_str.endswith("b"):
        return int(float(element_str[:-1]) * 1_000_000_000)
    else:
        return int(element_str)


def get_page(page_index: int):
    curr_page_url = f"{SUPPORTED_MODEL_NAME_PAGES_FORMAT}&p={page_index}"
    response = requests.get(curr_page_url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        return soup
    return None


def card_filter(model_card: Tag, model_name: str) -> bool:
    if is_blacklisted_model(model_name):
        return False
    if requires_additional_library(model_name):
        return False
    numeric_contents = get_numeric_contents(model_card)
    if len(numeric_contents) < 2:
        # If the model card doesn't have at least 2 numeric contents,
        # It means that he doesn't have any downloads/likes, so it's not a valid model card.
        return False
    number_of_downloads = convert_to_int(numeric_contents[0])
    if number_of_downloads < MIN_NUMBER_OF_DOWNLOADS:
        return False
    number_of_likes = convert_to_int(numeric_contents[1])
    if number_of_likes < MIN_NUMBER_OF_LIKES:
        return False
    return True


def get_model_names(soup: BeautifulSoup):
    model_cards: List[Tag] = soup.find_all("article", class_="overview-card-wrapper group", recursive=True)
    for model_card in model_cards:
        model_name = get_model_name(model_card)
        if card_filter(model_card, model_name):
            yield model_name


def generate_supported_model_names() -> Generator[str, None, None]:
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_index = {executor.submit(get_page, index): index for index in range(100)}
        for future in as_completed(future_to_index):
            soup = future.result()
            if soup:
                yield from get_model_names(soup)


def get_supported_model_names() -> Set[str]:
    cache_file_path = os.path.join(CURRENT_DIR, "supported_model_names.txt")
    if os.path.exists(cache_file_path):
        with open(cache_file_path, "r") as f:
            return set(f.read().splitlines())
    supported_model_names = set(generate_supported_model_names())
    with open(cache_file_path, "w") as f:
        f.write("\n".join(supported_model_names))
    return supported_model_names


def get_tested_model_names() -> Set[str]:
    return {
        "gpt2",
        "Gustavosta/MagicPrompt-Stable-Diffusion",
        "EleutherAI/gpt-neo-125M",
        "distilgpt2",
        "bigscience/bloom-560m",
        "bigscience/bloomz-560m",
        "microsoft/CodeGPT-small-py",
        "microsoft/DialoGPT-small",
    }


@pytest.mark.parametrize("model_name", get_tested_model_names())
def test_pipeline_creation(model_name: str):
    pipeline: GroupedSamplingPipeLine = GroupedSamplingPipeLine(
        model_name=model_name, group_size=5
    )
    assert pipeline is not None
    assert pipeline.model_name == model_name
    assert pipeline.wrapped_model.group_size == 5
    assert pipeline.wrapped_model.end_of_sentence_stop is True


def get_tokenizer_name(
        model_name: str,
) -> str:
    if model_name == "NovelAI/genji-jp":
        return "EleutherAI/gpt-j-6B"
    return model_name


@pytest.mark.parametrize("model_name", get_supported_model_names())
def test_supported_tokenizers(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(get_tokenizer_name(model_name))
    assert tokenizer is not None
    config = AutoConfig.from_pretrained(model_name)
    assert config is not None
    padding_id = get_padding_id(tokenizer)
    assert padding_id is not None
    assert isinstance(padding_id, int)
    assert padding_id >= 0
    end_of_sentence_id = get_end_of_text_id(tokenizer, config)
    assert end_of_sentence_id is not None
    assert isinstance(end_of_sentence_id, int)
    assert end_of_sentence_id >= 0

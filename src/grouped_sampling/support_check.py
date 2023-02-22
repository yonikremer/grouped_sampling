import os
from functools import lru_cache
from typing import Generator, Set, Union, List, Optional

import requests
from bs4 import BeautifulSoup, Tag, NavigableString, PageElement
from concurrent.futures import ThreadPoolExecutor, as_completed

from cache_to_disk import cache_to_disk

SUPPORTED_MODEL_NAME_PAGES_FORMAT = "https://huggingface.co/models?pipeline_tag=text-generation&library=pytorch"
BLACKLISTED_ORGANIZATIONS = {
    "huggingtweets",
}
DEFAULT_MIN_NUMBER_OF_DOWNLOADS = 50
DEFAULT_MIN_NUMBER_OF_LIKES = 5


@lru_cache(maxsize=1)
def get_unsupported_model_names() -> Set[str]:
    """returns the set of unsupported model names"""
    this_file_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(this_file_dir, "unsupported_models.txt"), "r") as f:
        return set(f.read().splitlines())


def get_model_name(model_card: Tag) -> str:
    """returns the model name from the model card tag"""
    h4_class = "text-md truncate font-mono text-black dark:group-hover:text-yellow-500 group-hover:text-indigo-600"
    h4_tag = model_card.find("h4", class_=h4_class)
    return h4_tag.text


def is_a_number(element: Union[PageElement, Tag]) -> bool:
    """returns True if the element is a number, False otherwise"""
    if isinstance(element, Tag):
        return False
    text = element.text
    lowered_text = text.strip().lower()
    no_characters_text = lowered_text.replace("k", "").replace("m", "").replace("b", "")
    element = no_characters_text.replace(",", "").replace(".", "")
    try:
        float(element)
    except ValueError:
        return False
    return True


def get_numeric_contents(model_card: Tag) -> List[PageElement]:
    """returns the number of likes and downloads from the model card tag it they exist in the model card"""
    div: Union[Tag | NavigableString] = model_card.find(
        "div",
        class_="mr-1 flex items-center overflow-hidden whitespace-nowrap text-sm leading-tight text-gray-400",
        recursive=True
    )
    contents: List[PageElement] = div.contents
    number_contents: List[PageElement] = [content for content in contents if is_a_number(content)]
    return number_contents


def convert_to_int(element: PageElement) -> int:
    """converts the element to an int"""
    element_str = element.text.strip().lower()
    if element_str.endswith("k"):
        return int(float(element_str[:-1]) * 1_000)
    elif element_str.endswith("m"):
        return int(float(element_str[:-1]) * 1_000_000)
    elif element_str.endswith("b"):
        return int(float(element_str[:-1]) * 1_000_000_000)
    return int(element_str)


def get_page(page_index: int) -> Optional[BeautifulSoup]:
    """returns the page with the given index if it exists, None otherwise"""
    curr_page_url = f"{SUPPORTED_MODEL_NAME_PAGES_FORMAT}&p={page_index}"
    response = requests.get(curr_page_url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        return soup
    return None


def has_a_model_card(model_name) -> bool:
    """returns True if the model card exists for the given model name, False otherwise"""
    model_card_url = f"https://huggingface.co/{model_name}"
    response = requests.get(model_card_url)
    if response.status_code == 200:
        return "no model card" not in response.text.lower()
    return False


def has_a_warning(model_name) -> bool:
    """returns True if the model card for the given model name, has a warning and False otherwise"""
    model_card_url = f"https://huggingface.co/{model_name}"
    response = requests.get(model_card_url)
    if response.status_code == 200:
        return "warning" in response.text.lower()
    return False


def card_filter(
        model_card: Tag,
        model_name: str,
        min_number_of_downloads: int,
        min_number_of_likes: int,
) -> bool:
    """returns True if the model card is valid, False otherwise"""
    if "8bit" in model_name:
        # 8bit models are not supported, U should use load_in_8bit=True instead
        return False
    if model_name in get_unsupported_model_names():
        return False
    organization = model_name.split("/")[0]
    if organization in BLACKLISTED_ORGANIZATIONS:
        return False
    if not has_a_model_card(model_name):
        return False
    if has_a_warning(model_name):
        return False
    numeric_contents = get_numeric_contents(model_card)
    if len(numeric_contents) < 2:
        # If the model card doesn't have at least 2 numeric contents,
        # It means that he doesn't have any downloads/likes, so it's not a valid model card.
        return False
    number_of_downloads = convert_to_int(numeric_contents[0])
    if number_of_downloads < min_number_of_downloads:
        return False
    number_of_likes = convert_to_int(numeric_contents[1])
    if number_of_likes < min_number_of_likes:
        return False
    return True


def get_model_names(
        soup: BeautifulSoup,
        min_number_of_downloads: int,
        min_number_of_likes: int,
) -> Generator[str, None, None]:
    """Scrapes the model names from the given soup"""
    model_cards: List[Tag] = soup.find_all("article", class_="overview-card-wrapper group", recursive=True)
    for model_card in model_cards:
        model_name = get_model_name(model_card)
        if card_filter(
                model_card=model_card,
                model_name=model_name,
                min_number_of_downloads=min_number_of_downloads,
                min_number_of_likes=min_number_of_likes
        ):
            yield model_name


def generate_supported_model_names(
        min_number_of_downloads: int,
        min_number_of_likes: int,
) -> Generator[str, None, None]:
    with ThreadPoolExecutor() as executor:
        future_to_index = {executor.submit(get_page, index): index for index in range(300)}
        for future in as_completed(future_to_index):
            soup = future.result()
            if soup:
                yield from get_model_names(
                    soup=soup,
                    min_number_of_downloads=min_number_of_downloads,
                    min_number_of_likes=min_number_of_likes,
                )


@cache_to_disk(n_days_to_cache=10)
def get_supported_model_names(
        min_number_of_downloads: int = DEFAULT_MIN_NUMBER_OF_DOWNLOADS,
        min_number_of_likes: int = DEFAULT_MIN_NUMBER_OF_LIKES,
) -> Set[str]:
    return set(
        generate_supported_model_names(
            min_number_of_downloads=min_number_of_downloads,
            min_number_of_likes=min_number_of_likes,
        )
    )


def is_supported(
        model_name: str,
        min_number_of_downloads: int = DEFAULT_MIN_NUMBER_OF_DOWNLOADS,
        min_number_of_likes: int = DEFAULT_MIN_NUMBER_OF_LIKES,
) -> bool:
    return model_name in get_supported_model_names(
        min_number_of_downloads=min_number_of_downloads,
        min_number_of_likes=min_number_of_likes,
    )


class UnsupportedModelNameException(Exception):
    def __init__(self, model_name: str):
        super().__init__(
            f"The model name {model_name} is not supported yet."
            f"The algorithm only supports models that appears here: {SUPPORTED_MODEL_NAME_PAGES_FORMAT}"
            f"with at least {DEFAULT_MIN_NUMBER_OF_DOWNLOADS} downloads and {DEFAULT_MIN_NUMBER_OF_LIKES} likes"
            f"That wasn't created by one of the following organizations: {BLACKLISTED_ORGANIZATIONS}"
            f"and that aren't one of the following models: {get_unsupported_model_names()}"
        )


def check_support(model_name: str) -> None:
    if not is_supported(model_name):
        raise UnsupportedModelNameException(model_name)


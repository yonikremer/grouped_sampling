import os
from functools import lru_cache, partial
from typing import Generator, Set, Union, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

from bs4 import BeautifulSoup, Tag, NavigableString, PageElement
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
    if "mt0" in model_name:
        # mt0 models are not supported, U they are sequence to sequence models and not causal language models
        # even though they are labeled as causal language models in the model card
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


def get_model_name(model_card: Tag) -> str:
    """returns the model name from the model card tag"""
    h4_class = "text-md truncate font-mono text-black dark:group-hover:text-yellow-500 group-hover:text-indigo-600"
    h4_tag = model_card.find("h4", class_=h4_class)
    return h4_tag.text


def filtered_get_model_name(
        model_card: Tag,
        min_number_of_downloads: int,
        min_number_of_likes: int,
):
    """returns the model name from the model card tag if it's valid, None otherwise"""
    model_name = get_model_name(model_card)
    if card_filter(
            model_card=model_card,
            model_name=model_name,
            min_number_of_downloads=min_number_of_downloads,
            min_number_of_likes=min_number_of_likes
    ):
        return model_name
    return None


def get_model_names(
        soup: BeautifulSoup,
        min_number_of_downloads: int,
        min_number_of_likes: int,
) -> Generator[str, None, None]:
    """Scrapes the model names from the given soup"""
    model_cards: List[Tag] = soup.find_all("article", class_="overview-card-wrapper group", recursive=True)
    executor: ThreadPoolExecutor
    filtered_get_model_name_partial = partial(
        filtered_get_model_name,
        min_number_of_downloads=min_number_of_downloads,
        min_number_of_likes=min_number_of_likes,
    )
    with ThreadPoolExecutor() as executor:
        model_name_futures = (
            executor.submit(filtered_get_model_name_partial, model_card) for model_card in model_cards
        )
        for future in as_completed(model_name_futures):
            model_name = future.result()
            if model_name:
                print(model_name)
                yield model_name


def scrape_supported_model_names_in_page(
        page_index: int,
        min_number_of_downloads: int,
        min_number_of_likes: int,
) -> Generator[str, None, None]:
    """returns the list of supported model names"""
    page: Optional[BeautifulSoup] = get_page(page_index)
    if page:
        page: BeautifulSoup
        yield from get_model_names(
            soup=page,
            min_number_of_downloads=min_number_of_downloads,
            min_number_of_likes=min_number_of_likes,
        )


def generate_supported_model_names(
        min_number_of_downloads: int,
        min_number_of_likes: int,
) -> Generator[str, None, None]:
    scrape_supported_model_names_in_page_partial = partial(
        scrape_supported_model_names_in_page,
        min_number_of_downloads=min_number_of_downloads,
        min_number_of_likes=min_number_of_likes,
    )
    with ThreadPoolExecutor() as executor:
        future_to_index = {
            executor.submit(
                scrape_supported_model_names_in_page_partial,
                index
            ): index for index in range(300)
        }
        for future in as_completed(future_to_index):
            yield from future.result()


@cache_to_disk(n_days_to_cache=10)
def get_all_supported_model_names_set(
        min_number_of_downloads: int = DEFAULT_MIN_NUMBER_OF_DOWNLOADS,
        min_number_of_likes: int = DEFAULT_MIN_NUMBER_OF_LIKES,
) -> Set[str]:
    return set(
        generate_supported_model_names(
            min_number_of_downloads=min_number_of_downloads,
            min_number_of_likes=min_number_of_likes,
        )
    )


def unsupported_model_name_error_message(model_name: str) -> str:
    return \
        f"""{model_name} is not supported yet.
        The algorithm only supports models that appear here: {SUPPORTED_MODEL_NAME_PAGES_FORMAT}.
        with at least {DEFAULT_MIN_NUMBER_OF_DOWNLOADS} downloads and {DEFAULT_MIN_NUMBER_OF_LIKES} likes.
        That wasn't created by one of the following organizations: {BLACKLISTED_ORGANIZATIONS}.
        And that aren't one of the following models: {get_unsupported_model_names()}.
        And that have a non empty model card without a waring.
        And they are not ab mt0 model or an 8bit model."""


def is_supported(
        model_name: str,
        min_number_of_downloads: int = DEFAULT_MIN_NUMBER_OF_DOWNLOADS,
        min_number_of_likes: int = DEFAULT_MIN_NUMBER_OF_LIKES,
) -> bool:
    return model_name in get_all_supported_model_names_set(min_number_of_downloads=min_number_of_downloads,
                                                           min_number_of_likes=min_number_of_likes)


class UnsupportedModelNameException(Exception):
    def __init__(self, model_name: str):
        super().__init__(
            unsupported_model_name_error_message(model_name)
        )


def check_support(model_name: str) -> None:
    if not is_supported(model_name):
        raise UnsupportedModelNameException(model_name)


def main() -> None:
    for i, model_name in enumerate(
            generate_supported_model_names(
                min_number_of_downloads=0,
                min_number_of_likes=0,
            )
    ):
        print(f"{i}: {model_name}")


if __name__ == "__main__":
    print(unsupported_model_name_error_message("test"))

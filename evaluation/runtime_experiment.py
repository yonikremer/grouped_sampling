import json
import time
from os.path import abspath, dirname, join
from typing import List, Iterable, Dict

from datasets import get_dataset_config_names, Dataset
import matplotlib.pyplot as plt

from evaluation import process_translation_data, DATASET_NAME, lang_code_to_name, create_pipeline

curr_dir = dirname(abspath(__file__))


def prompt_engineering(prompts: Iterable[str], input_language_code: str, output_language_code: str):
    input_language = lang_code_to_name(input_language_code)
    output_language = lang_code_to_name(output_language_code)
    return [f"Translate {input_language} to {output_language}.\n{input_language}: {prompt} \n{output_language}: "
            for prompt in prompts]


def get_prompts(debug: bool) -> List[str]:
    """Gets a list of all the examples in the dataset with name DATASET_NAME"""
    sub_set_names = get_dataset_config_names(DATASET_NAME)
    if debug:
        sub_set_names = sub_set_names[:1]
    prompts: List[str] = []
    subset_part1: Dataset
    subset_part2: Dataset
    language_code1: str
    language_code2: str
    for sub_set_name in sub_set_names:
        subset_part, _, language_code1, language_code2 = process_translation_data(sub_set_name, debug)
        prompts.extend(subset_part[language_code1])
        prompts.extend(subset_part[language_code2])
    return prompts


def create_and_save_graph(group_size_to_duration: Dict[int, float]):
    curr_figure = plt.gcf()
    curr_figure.clear()
    plt.scatter(
        group_size_to_duration.keys(),
        group_size_to_duration.values(),
    )
    plt.xlabel("Group Size")
    plt.ylabel("Duration (Hours)")
    plt.xscale("log")
    title = "Duration as a function of (log scaled) Group Size"
    plt.title(title)
    PLOTS_FOLDER = join(curr_dir, "plots")
    fig_path = join(PLOTS_FOLDER, join(PLOTS_FOLDER, f"{title}.png"))
    print(f"Saving figure to {fig_path}")
    plt.savefig(fig_path, bbox_inches='tight', pad_inches=0)


def change_group_size(new_group_size: int):
    """Changes the group size value
     in the hyper_params_file
      at ./final_project/evaluation/evaluated_text_generator_dict.json"""
    hyper_params_file = join(curr_dir, "evaluated_text_generator_dict.json")
    with open(hyper_params_file, 'r+') as f:
        data = json.load(f)
        data['group_size'] = new_group_size
        f.seek(0)
        json.dump(data, f, indent=4)
        f.truncate()


def main(debug: bool = False):
    if debug:
        group_sizes = [1024]
    else:
        group_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    group_size_to_duration = {}
    prompts: List[str] = get_prompts(debug=debug)
    for group_size in group_sizes:
        change_group_size(group_size)
        pipeline = create_pipeline()
        start_time = time.time()
        pipeline(prompts)
        end_time = time.time()
        duration_seconds = end_time - start_time
        group_size_to_duration[group_size] = duration_seconds / 3600
    create_and_save_graph(group_size_to_duration)


if __name__ == "__main__":
    main(False)

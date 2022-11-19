from __future__ import annotations

from typing import Generator, Any, Dict, Tuple, List

from evaluate import TranslationEvaluator
from datasets import load_dataset, Dataset, get_dataset_config_names

from evaluation.experiment_manager import ExperimentManager
from evaluation.helpers import lang_code_to_name
from sampling_generator import SamplingGenerator
from text_generator import TextGenerator

from transformers.utils.logging import disable_progress_bar

disable_progress_bar()

from datasets.utils.logging import disable_progress_bar

disable_progress_bar()

DATASET_NAME = "ted_talks_iwslt"
METRIC_NAME = "bertscore"


def generate_text_generators() -> Generator[TextGenerator, None, None]:
    yield SamplingGenerator(
        model_name="facebook/opt-125m",
        group_size=4,
        temp=1.0,
        top_k=None,
        top_p=1.0,
        end_of_sentence_stop=True,
        answer_length_multiplier=2,
    )


def process_translation_data(sub_set_name: str) -> Tuple[Dataset, Dataset, str, str]:
    spited_sub_set_name = sub_set_name.split("_")
    language_code1, language_code2 = spited_sub_set_name[:2]
    # noinspection PyUnreachableCode
    if __debug__:
        sub_set = load_dataset(DATASET_NAME, sub_set_name, split="train[:2]")
    else:
        sub_set = load_dataset(DATASET_NAME, sub_set_name, split="train")
    processed_data1_dict: Dataset
    processed_data2_dict: Dataset

    def rename_keys(x: Dict[str, Any], input_lang_name: str, output_lang_name: str) -> Dict[str, str]:
        translation: Dict[str, str] = x["translation"]
        return {input_lang_name: translation[input_lang_name], output_lang_name: translation[output_lang_name]}

    subset_part1 = sub_set.map(rename_keys,
                               fn_kwargs={"input_lang_name": language_code1, "output_lang_name": language_code2})
    subset_part2 = sub_set.map(rename_keys,
                               fn_kwargs={"input_lang_name": language_code2, "output_lang_name": language_code1})
    return subset_part1, subset_part2, language_code1, language_code2


def sub_experiment_half(
        my_evaluator: TranslationEvaluator,
        sub_set_half: Dataset,
        in_lang_code: str, out_lang_code: str,
        generator: TextGenerator,
        manager: ExperimentManager) -> None:
    input_lang_name, output_lang_name = lang_code_to_name(in_lang_code), lang_code_to_name(out_lang_code)
    prefix = f"Translate {input_lang_name} to {output_lang_name}: \n {input_lang_name}: "
    postfix = f"\n {output_lang_name}: "
    my_evaluator.METRIC_KWARGS = {"lang": out_lang_code}
    my_evaluator.PIPELINE_KWARGS = {"prefix": prefix, "postfix": postfix}
    # noinspection PyTypeChecker
    scores: Dict[str, List[float] | Any] = my_evaluator.compute(
        model_or_pipeline=generator,
        data=sub_set_half,
        input_column=in_lang_code,
        label_column=out_lang_code
    )
    manager.log_sub_experiment(scores, in_lang_code, out_lang_code, sub_set_half)


def run_experiment(generator: TextGenerator) -> None:
    generator.task = "translation"
    my_evaluator = TranslationEvaluator(default_metric_name=METRIC_NAME)
    my_evaluator.PREDICTION_PREFIX = "generated"
    manager = ExperimentManager(generator)
    sub_sut_names = get_dataset_config_names(DATASET_NAME)
    for i, sub_set_name in enumerate(sub_sut_names):
        print(f"Running sub-experiment {i + 1} out of {len(sub_sut_names)}")
        processed_sub_set: Dataset
        language_code1: str
        language_code2: str
        subset_part1, subset_part2, language_code1, language_code2 = process_translation_data(sub_set_name)
        sub_experiment_half(my_evaluator, subset_part1, language_code1, language_code2, generator, manager)
        sub_experiment_half(my_evaluator, subset_part2, language_code2, language_code1, generator, manager)
    manager.end_experiment()


def main() -> None:
    for curr_text_generator in generate_text_generators():
        run_experiment(curr_text_generator)


if __name__ == "__main__":
    main()

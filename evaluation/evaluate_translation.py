from __future__ import annotations

from typing import Any, Dict, List
from warnings import warn

from datasets import Dataset, get_dataset_config_names
from evaluate import TranslationEvaluator

from evaluation import (
    DATASET_NAME,
    create_pipeline,
    disable_progress_bars,
    lang_code_to_name,
    process_translation_data,
)
from evaluation.experiment_manager import ExperimentManager
from src.grouped_sampling import GroupedGenerationPipeLine

disable_progress_bars()


def sub_experiment_half(
    my_evaluator: TranslationEvaluator,
    sub_set_half: Dataset,
    in_lang_code: str,
    out_lang_code: str,
    generator: GroupedGenerationPipeLine,
    manager: ExperimentManager,
) -> None:
    input_lang_name, output_lang_name = lang_code_to_name(
        in_lang_code), lang_code_to_name(out_lang_code)
    prefix = (
        f"Translate {input_lang_name} to {output_lang_name}: \n {input_lang_name}: "
    )
    postfix = f"\n {output_lang_name}: "
    my_evaluator.METRIC_KWARGS = {"lang": out_lang_code}
    my_evaluator.PIPELINE_KWARGS = {"prefix": prefix, "postfix": postfix}
    # noinspection PyTypeChecker
    scores: Dict[str, List[float] | Any] = my_evaluator.compute(
        model_or_pipeline=generator,
        data=sub_set_half,
        input_column=in_lang_code,
        label_column=out_lang_code,
    )
    manager.log_sub_experiment(scores, in_lang_code, out_lang_code,
                               sub_set_half)


def run_experiment(
    pipeline: GroupedGenerationPipeLine,
    my_evaluator: TranslationEvaluator,
    sub_sut_names: List[str],
    debug: bool,
) -> None:
    pipeline.task = "translation"
    manager = ExperimentManager(
        debug=debug,
        parameters=pipeline.as_dict(),
    )
    for i, sub_set_name in enumerate(sub_sut_names):
        print(f"Running sub-experiment {i + 1} out of {len(sub_sut_names)}")
        subset_part1: Dataset
        subset_part2: Dataset
        language_code1: str
        language_code2: str
        (
            subset_part1,
            subset_part2,
            language_code1,
            language_code2,
        ) = process_translation_data(sub_set_name, debug)
        sub_experiment_half(
            my_evaluator,
            subset_part1,
            language_code1,
            language_code2,
            pipeline,
            manager,
        )
        if not debug:
            sub_experiment_half(
                my_evaluator,
                subset_part2,
                language_code2,
                language_code1,
                pipeline,
                manager,
            )
    manager.end_experiment()


def create_evaluator() -> TranslationEvaluator:
    """Creates a translation evaluator"""
    METRIC_NAME = "bertscore"
    my_evaluator = TranslationEvaluator(default_metric_name=METRIC_NAME)
    my_evaluator.PREDICTION_PREFIX = "generated"
    return my_evaluator


def main(debug: bool = __debug__) -> None:
    if debug:
        # send a warning
        warn(
            "Running in debug mode, only a small subset of the data will be used"
        )
    sub_sut_names = get_dataset_config_names(DATASET_NAME)
    if debug:
        sub_sut_names = sub_sut_names[:1]
    curr_text_generator = create_pipeline()
    curr_evaluator = create_evaluator()
    run_experiment(curr_text_generator, curr_evaluator, sub_sut_names, debug)


if __name__ == "__main__":
    main()

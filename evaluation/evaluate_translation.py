from typing import Generator, Any, Dict, Tuple, List

from evaluate import TranslationEvaluator
from datasets import load_dataset, Dataset, get_dataset_config_names

from evaluation.experiment_manager import ExperimentManager
from sampling_generator import SamplingGenerator
from text_generator import TextGenerator

DATASET_NAME = "ted_talks_iwslt"


# the next line is used for debug mode
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def generate_text_generators() -> Generator[TextGenerator, None, None]:
    yield SamplingGenerator(
        model_name="facebook/opt-125m",
        group_size=16,
        temp=1.0,
        top_k=None,
        top_p=1.0,
        end_of_sentence_stop=True,
        answer_length_multiplier=2,
    )


def process_translation_data(data_set_name: str, sub_set_name: str) -> Tuple[Dataset, str, str]:
    spited_sub_set_name = sub_set_name.split("_")
    language1, language2 = spited_sub_set_name[:2]
    # add a warning here
    sub_set: Dataset = load_dataset(data_set_name, sub_set_name, split="train")
    processed_data1_dict: Dataset
    processed_data2_dict: Dataset

    def rename_keys(x: Dict[str, Any], input_lang: str, output_lang: str) -> Dict[str, str]:
        translation: Dict[str, str] = x["translation"]
        return {input_lang: translation[input_lang], output_lang: translation[output_lang]}

    processed_data = sub_set.map(rename_keys, fn_kwargs={"input_lang": language1, "output_lang": language2})
    return processed_data, language1, language2


def run_experiment(generator: TextGenerator) -> None:
    generator.task = "translation"
    my_evaluator = TranslationEvaluator(default_metric_name="bertscore")
    my_evaluator.PREDICTION_PREFIX = "generated"
    manager = ExperimentManager(generator)
    for sub_set_name in get_dataset_config_names(DATASET_NAME):
        processed_sub_set: Dataset
        language1: str
        language2: str
        processed_sub_set, language1, language2 = process_translation_data(DATASET_NAME, sub_set_name)
        my_evaluator.METRIC_KWARGS = {"lang": language2}
        # noinspection PyTypeChecker
        scores1: Dict[str, List[float]] = my_evaluator.compute(
            model_or_pipeline=generator,
            data=processed_sub_set,
            input_column=language1,
            label_column=language2
        )
        manager.log_sub_experiment(scores1)
        my_evaluator.METRIC_KWARGS = {"lang": language1}
        # noinspection PyTypeChecker
        scores2: Dict[str, List[float]] = my_evaluator.compute(
            model_or_pipeline=generator,
            data=processed_sub_set,
            input_column=language2,
            label_column=language1
        )
        manager.log_sub_experiment(scores2)
    manager.end_experiment()


def main():
    for curr_text_generator in generate_text_generators():
        run_experiment(curr_text_generator)


if __name__ == "__main__":
    main()

import os
from typing import Generator, Any, Dict, List
from datetime import datetime

from evaluate import TranslationEvaluator
from datasets import load_dataset, Dataset

from sampling_generator import SamplingGenerator
from text_generator import TextGenerator
from comet_ml import Experiment

DATASET_NAME = "news_commentary"
SUB_SETS = ['ar-cs', 'ar-de', 'cs-de', 'ar-en', 'cs-en', 'de-en', 'ar-es', 'cs-es', 'de-es', 'en-es', 'ar-fr', 'cs-fr',
            'de-fr', 'en-fr', 'es-fr', 'ar-it', 'cs-it', 'de-it', 'en-it', 'es-it', 'fr-it', 'ar-ja', 'cs-ja', 'de-ja',
            'en-ja', 'es-ja', 'fr-ja', 'ar-nl', 'cs-nl', 'de-nl', 'en-nl', 'es-nl', 'fr-nl', 'it-nl', 'ar-pt', 'cs-pt',
            'de-pt', 'en-pt', 'es-pt', 'fr-pt', 'it-pt', 'nl-pt', 'ar-ru', 'cs-ru', 'de-ru', 'en-ru', 'es-ru', 'fr-ru',
            'it-ru', 'ja-ru', 'nl-ru', 'pt-ru', 'ar-zh', 'cs-zh', 'de-zh', 'en-zh', 'es-zh', 'fr-zh', 'it-zh', 'ja-zh',
            'nl-zh', 'pt-zh', 'ru-zh']
METRIC_NAMES = ("bertscore",)
if os.path.exists("comet_ml_api_key.txt"):
    with open("comet_ml_api_key.txt", "r") as f:
        COMET_ML_API_KEY = f.read().strip()
else:
    COMET_ML_API_KEY = input("Please enter your api_key for comet ml: ")
PROJECT_NAME = "grouped-sampling-evaluation"


class BertScoresHandler:
    per_example_f1: List[float] = []
    per_example_precision: List[float] = []
    per_example_recall: List[float] = []
    num_examples: int = 0
    start_time: datetime

    def __init__(self):
        self.start_time = datetime.now()

    def log_sub_experiment(self, bert_scores) -> None:
        self.per_example_f1.extend(bert_scores["f1"].tolist())
        self.per_example_precision.extend(bert_scores["precision"].tolist())
        self.per_example_recall.extend(bert_scores["recall"].tolist())
        self.num_examples += len(bert_scores["f1"])

    def end_experiment(self, experiment: Experiment) -> None:
        average_bert_f1 = sum(self.per_example_f1) / self.num_examples
        average_bert_precision = sum(self.per_example_precision) / self.num_examples
        average_bert_recall = sum(self.per_example_recall) / self.num_examples
        experiment.log_metric("average_bert_f1", average_bert_f1)
        experiment.log_metric("average_bert_precision", average_bert_precision)
        experiment.log_metric("average_bert_recall", average_bert_recall)
        total_time_in_seconds = (datetime.now() - self.start_time).total_seconds()
        experiment.log_metric("time in seconds", total_time_in_seconds)
        experiment.send_notification(f"Experiment finished successfully in {total_time_in_seconds} seconds")
        experiment.end()


def generate_text_generators() -> Generator[TextGenerator, None, None]:
    yield SamplingGenerator(
        model_name="facebook/opt-125m",
        group_size=8,
        temp=1.0,
        top_k=3,
        top_p=None,
        end_of_sentence_stop=True,
    )


def process_translation_data(data_set_name: str, sub_set_name: str) -> Dataset:
    language1, language2 = sub_set_name.split("-")
    # add a warning here
    sub_set: Dataset = load_dataset(data_set_name, sub_set_name, split="train")
    processed_data1_dict: Dataset
    processed_data2_dict: Dataset

    def rename_keys(x: Dict[str, Any], input_lang: str, output_lang: str) -> Dict[str, str]:
        translation: Dict[str, str] = x["translation"]
        return {input_lang: translation[input_lang], output_lang: translation[output_lang]}

    processed_data = sub_set.map(rename_keys, fn_kwargs={"input_lang": language1, "output_lang": language2})
    return processed_data


def run_experiment(generator: TextGenerator) -> None:
    generator.task = "translation"
    experiment = Experiment(api_key=COMET_ML_API_KEY, project_name=PROJECT_NAME)
    experiment.log_parameters(generator.to_dict())
    my_evaluator = TranslationEvaluator(default_metric_name="bertscore")
    my_evaluator.PREDICTION_PREFIX = "generated"
    num_samples = 0
    handler = BertScoresHandler()
    for sub_set_name in SUB_SETS:
        language1, language2 = sub_set_name.split("-")
        processed_sub_set: Dataset = process_translation_data(DATASET_NAME, sub_set_name)
        num_samples += len(processed_sub_set)
        my_evaluator.METRIC_KWARGS = {"lang": language2}
        scores1 = my_evaluator.compute(model_or_pipeline=generator,
                                       data=processed_sub_set, input_column=language1, label_column=language2)
        handler.log_sub_experiment(scores1)
        my_evaluator.METRIC_KWARGS = {"lang": language1}
        scores2 = my_evaluator.compute(model_or_pipeline=generator,
                                       data=processed_sub_set, input_column=language2, label_column=language1)
        handler.log_sub_experiment(scores2)
    handler.end_experiment(experiment)


def main():
    for curr_text_generator in generate_text_generators():
        print("starting experiment")
        run_experiment(curr_text_generator)


if __name__ == "__main__":
    main()

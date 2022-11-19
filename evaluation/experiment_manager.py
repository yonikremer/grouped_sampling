from __future__ import annotations

import os
from datetime import datetime
from typing import List, Dict, Any, Callable, Tuple, Sequence, Set

from comet_ml import Experiment
from pandas import DataFrame, concat, Series
import matplotlib.pyplot as plt
from datasets import Dataset

from evaluation.helpers import lang_code_to_name

try:
    # noinspection PyUnresolvedReferences
    from kaggle_secrets import UserSecretsClient
except ImportError:
    using_kaggle = False
else:  # if we are using kaggle, we need to set the api key
    using_kaggle = True

from text_generator import TextGenerator


class ExperimentManager:
    start_time: datetime
    experiment: Experiment
    COMET_ML_PROJECT_NAME = "grouped-sampling-evaluation"
    df: DataFrame
    language_pairs: Set[Tuple[str, str]] = set()

    def __init__(self, generator: TextGenerator):
        self.experiment = Experiment(
            api_key=ExperimentManager.get_comet_api_key(),
            project_name=self.COMET_ML_PROJECT_NAME,
            auto_param_logging=False,
            auto_metric_logging=False,
            log_code=False,
            log_git_metadata=False,
            log_git_patch=False,
        )
        self.experiment.log_parameters(generator.as_dict())
        self.start_time = datetime.now()
        self.df = DataFrame(columns=[
            "input_text",
            "target_text",
            "input_language",
            "output_language",
            "BERT_f1",
            "BERT_precision",
            "BERT_recall"
        ])

    @staticmethod
    def get_comet_api_key() -> str:
        """Returns the Comet API key from the file "final_project/evaluate/comet_api_key.txt"
        if this file does not exist, asks the user to enter the key manually and saves it to the file"""
        if using_kaggle:
            return UserSecretsClient().get_secret("comet_ml_api_key")
        if os.getcwd() == "/content":
            # if running on colab
            api_key_file = "final_project/evaluation/comet_ml_api_key.txt"
        else:
            # if running locally
            api_key_file = "comet_ml_api_key.txt"
        if os.path.exists(api_key_file):
            with open(api_key_file, "r") as f:
                return f.read().strip()
        api_key = input("Please enter your api_key for comet ml: ")
        with open(api_key_file, "w") as f:
            f.write(api_key)
        return api_key

    def log_stats(self, scores: DataFrame, title: str) -> None:
        STAT_NAME_TO_FUNC: Sequence[Tuple[str, Callable[[Series], float]]] = (
            ("mean", lambda x: x.mean()),
            ("standard_deviation", lambda x: x.std()),
            ("min", lambda x: x.min()),
            ("max", lambda x: x.max()),
            ("median", lambda x: x.median()),
            ("25_percentile", lambda x: x.quantile(0.25)),
            ("75_percentile", lambda x: x.quantile(0.75)),
        )
        for score_name in ("BERT_f1", "BERT_precision", "BERT_recall"):
            curr_column = scores[score_name]
            score_stats = {f"{title}_{score_name}_{stat_name}": stat_func(curr_column)
                           for stat_name, stat_func in STAT_NAME_TO_FUNC}
            self.experiment.log_metrics(score_stats)
            plt.hist(curr_column, bins=20)
            plt.title(f"Histogram of {title}_{score_name}")
            self.experiment.log_figure(figure_name=f"{title}_{score_name}_histogram", figure=plt)

    def log_sub_experiment(
            self,
            bert_scores: Dict[str, List[float] | Any],
            input_lang_code: str,
            output_lang_code: str,
            sub_set: Dataset,
    ) -> None:
        """Args:
            bert_scores: Dict[str, Tensor]
                with keys "f1", "precision", "recall"
                values of shape (number of examples in the sub-experiment,) and type float
            input_lang_code: The name of the input language in this sub experiment half
            output_lang_code: The name of the output language in this sub experiment half
            sub_set: The dataset that was used for this sub experiment half"""
        input_lang_name, output_lang_name = lang_code_to_name(input_lang_code), lang_code_to_name(output_lang_code)
        self.language_pairs.add((input_lang_name, output_lang_name))
        f_1: List[float] = bert_scores["f1"]
        precision: List[float] = bert_scores["precision"]
        recall: List[float] = bert_scores["recall"]
        assert len(f_1) == len(precision) == len(recall)
        # add scores to the dataframe
        new_data: DataFrame = DataFrame.from_dict({
            "input_language": [input_lang_name] * len(f_1),
            "output_language": [output_lang_name] * len(f_1),
            "BERT_f1": f_1,
            "BERT_precision": precision,
            "BERT_recall": recall,
            "input_text": sub_set[input_lang_code],
            "target_text": sub_set[output_lang_code],
        })
        self.df = concat([self.df, new_data], ignore_index=True, copy=False)

    def end_experiment(self) -> None:
        """Logs the experiment to comet ml"""
        self.experiment.log_dataframe_profile(self.df, "BERT_scores", header=True)
        self.log_stats(self.df, "general")
        for input_lang, output_lang in self.language_pairs:
            pair_scores: DataFrame
            pair_scores = self.df[
                (self.df["input_lang_name"] == input_lang) & (self.df["output_language"] == output_lang)
                ]
            self.log_stats(pair_scores, f"{input_lang} to {output_lang}")
        total_time_in_seconds = (datetime.now() - self.start_time).total_seconds()
        num_examples = len(self.df)
        total_time_in_hours = total_time_in_seconds / 3600
        self.experiment.log_metrics({
            "time in hours": total_time_in_hours,
            "seconds per example": total_time_in_seconds / num_examples,
            "examples per second": num_examples / total_time_in_seconds
        })
        self.experiment.send_notification(f"Experiment finished successfully in {total_time_in_hours} hours")
        self.experiment.end()

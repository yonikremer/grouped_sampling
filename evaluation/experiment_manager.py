from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Set, Tuple

import matplotlib.pyplot as plt
from comet_ml import Experiment
from datasets import Dataset
from pandas import DataFrame, concat

from evaluation import (
    BERT_SCORES,
    STAT_NAME_TO_FUNC,
    get_comet_api_key,
    get_project_name,
    lang_code_to_name,
)


class ExperimentManager:
    start_time: datetime
    experiment: Experiment
    df: DataFrame
    language_pairs: Set[Tuple[str, str]] = set()

    def __init__(self, debug: bool, parameters: Dict[str, Any] = None):
        self.experiment = Experiment(
            api_key=get_comet_api_key(),
            project_name=get_project_name(debug=debug),
            auto_param_logging=False,
            auto_metric_logging=False,
            log_code=False,
            log_git_metadata=False,
            log_git_patch=False,
        )
        self.experiment.log_parameters(parameters)
        self.start_time = datetime.now()
        self.df = DataFrame(columns=[
            "input_text",
            "target_text",
            "input_language",
            "output_language",
        ] + list(BERT_SCORES))

    def log_stats(self, scores: DataFrame, title: str) -> None:
        self.experiment.log_dataframe_profile(scores,
                                              f"{title}_BERT_scores",
                                              header=True)
        for score_name in BERT_SCORES:
            curr_column = scores[score_name]
            score_stats = {
                f"{title}_{score_name}_{stat_name}": stat_func(curr_column)
                for stat_name, stat_func in STAT_NAME_TO_FUNC
            }
            self.experiment.log_metrics(score_stats)
            plt.hist(curr_column, bins=20)
            plt.title(f"Histogram of {title}_{score_name}")
            plt.legend()
            self.experiment.log_figure(
                figure_name=f"{title}_{score_name}_histogram", figure=plt)

    def log_sub_experiment(
        self,
        bert_scores: Dict[str, List[float] | Any],
        input_lang_code: str,
        output_lang_code: str,
        sub_set: Dataset,
    ) -> None:
        """
        Args:
        bert_scores: Dict[str, List[float] | Any]
            with keys "f1", "precision", "recall"
            values of shape (number of examples in the sub-experiment,) and type float
        input_lang_code: The name of the input language in this sub experiment half
        output_lang_code: The name of the output language in this sub experiment half
        sub_set: The dataset that was used for this sub experiment half
        """
        input_lang_name, output_lang_name = lang_code_to_name(
            input_lang_code), lang_code_to_name(output_lang_code)
        self.language_pairs.add((input_lang_name, output_lang_name))
        f_1: List[float] = bert_scores["f1"]
        precision: List[float] = bert_scores["precision"]
        recall: List[float] = bert_scores["recall"]
        if not len(f_1) == len(precision) == len(recall):
            raise AssertionError
        # add scores to the dataframe
        new_data: DataFrame = DataFrame.from_dict({
            "input_language": [input_lang_name] * len(f_1),
            "output_language": [output_lang_name] * len(f_1),
            "BERT_f1":
            f_1,
            "BERT_precision":
            precision,
            "BERT_recall":
            recall,
            "input_text":
            sub_set[input_lang_code],
            "target_text":
            sub_set[output_lang_code],
        })
        self.df = concat([self.df, new_data], ignore_index=True, copy=False)
        curr_time_diff = datetime.now() - self.start_time
        print(f"Translated {len(self.df)} examples in {curr_time_diff}")

    def end_experiment(self) -> None:
        """Logs the experiment to comet ml"""
        self.log_stats(self.df, "general")
        for input_lang, output_lang in self.language_pairs:
            pair_scores: DataFrame
            pair_scores = self.df[(self.df["input_language"] == input_lang)
                                  &
                                  (self.df["output_language"] == output_lang)]
            self.log_stats(pair_scores, f"{input_lang} to {output_lang}")
        total_time_in_seconds = (datetime.now() -
                                 self.start_time).total_seconds()
        num_examples = len(self.df)
        total_time_in_hours = total_time_in_seconds / 3600
        self.experiment.log_metrics({
            "time in hours":
            total_time_in_hours,
            "seconds per example":
            total_time_in_seconds / num_examples,
            "examples per second":
            num_examples / total_time_in_seconds,
        })
        self.experiment.send_notification(
            f"Experiment finished successfully in {total_time_in_hours} hours")
        self.experiment.end()

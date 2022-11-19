from __future__ import annotations

import os
from datetime import datetime
from typing import List, Dict, Any

from comet_ml import Experiment
from pandas import DataFrame
import matplotlib.pyplot as plt

try:
    # noinspection PyUnresolvedReferences
    from kaggle_secrets import UserSecretsClient
except ImportError:
    using_kaggle = False
else:  # if we are using kaggle, we need to set the api key
    using_kaggle = True

from text_generator import TextGenerator


class ExperimentManager:
    per_example_f1: List[float] = []
    per_example_precision: List[float] = []
    per_example_recall: List[float] = []
    start_time: datetime
    experiment: Experiment
    COMET_ML_PROJECT_NAME = "grouped-sampling-evaluation"
    df: DataFrame

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
        self.df = DataFrame(columns=["BERT_f1", "BERT_precision", "BERT_recall"], dtype=float)

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

    def log_sub_experiment(self, bert_scores: Dict[str, List[float] | Any]) -> None:
        """Args:
            bert_scores: Dict[str, Tensor]
                with keys "f1", "precision", "recall"
                values of shape (number of examples in the sub-experiment,) and type float"""
        f_1: List[float] = bert_scores["f1"]
        precision: List[float] = bert_scores["precision"]
        recall: List[float] = bert_scores["recall"]
        assert len(f_1) == len(precision) == len(recall)
        # add scores to the dataframe
        for curr_f1, curr_precision, curr_recall in zip(f_1, precision, recall):
            self.df = self.df.append({"BERT_f1": curr_f1,
                                      "BERT_precision": curr_precision,
                                      "BERT_recall": curr_recall}, ignore_index=True)

    def end_experiment(self) -> None:
        """Logs the experiment to comet ml"""
        self.experiment.log_dataframe_profile(self.df, "BERT_scores", header=True)
        for score_name in ("BERT_f1", "BERT_precision", "BERT_recall"):
            curr_column = self.df[score_name]
            score_stats = {
                f"{score_name}_mean": curr_column.mean(),
                f"{score_name}_std": curr_column.std(),
                f"{score_name}_min": curr_column.min(),
                f"{score_name}_max": curr_column.max(),
                f"{score_name}_median": curr_column.median(),
                f"{score_name}_25_percentile": curr_column.quantile(0.25),
                f"{score_name}_75_percentile": curr_column.quantile(0.75),
            }
            self.experiment.log_metrics(score_stats)
            plt.hist(curr_column, bins=20)
            plt.title(f"Histogram of {score_name}")
            self.experiment.log_figure(figure_name=f"{score_name}_histogram", figure=plt)
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

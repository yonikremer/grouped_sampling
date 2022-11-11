import os
from datetime import datetime
from typing import List, Dict

from comet_ml import Experiment

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
    num_examples: int = 0
    start_time: datetime
    experiment: Experiment
    COMET_ML_PROJECT_NAME = "grouped-sampling-evaluation"

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

    def log_sub_experiment(self, bert_scores: Dict[str, List[float]]) -> None:
        """Args:
            bert_scores: Dict[str, Tensor]
                with keys "f1", "precision", "recall"
                values of shape (number of examples in the sub-experiment,) and type float"""
        self.per_example_f1.extend(bert_scores["f1"])
        self.per_example_precision.extend(bert_scores["precision"])
        self.per_example_recall.extend(bert_scores["recall"])
        self.num_examples += len(bert_scores["f1"])

    def end_experiment(self) -> None:
        per_example_scores: Dict[str, List[float]] = {"f1": self.per_example_f1,
                                                      "precision": self.per_example_precision,
                                                      "recall": self.per_example_recall}
        averaged_scores = {}
        for metric, scores in per_example_scores.items():
            averaged_scores[f"average BERT {metric}"] = sum(scores) / self.num_examples
        self.experiment.log_metrics(averaged_scores)
        total_time_in_hours = (datetime.now() - self.start_time).total_seconds() / 3600
        self.experiment.log_metric("time in hours", total_time_in_hours)
        self.experiment.send_notification(f"Experiment finished successfully in {total_time_in_hours} hours")
        self.experiment.end()

# Evaluation

This directory contains the evaluation scripts for the project.

# Evaluation Process OverView:

I let the GroupedSamplingPipeLine translate all the talks from the ted_talks_iwslt dataset.
Then I use the `evaluation/evaluatate_translation.py` script to calculate the BERT scores of the translated talks.
The results are saved in [the comet ML project](https://www.comet.com/yonikremer/grouped-sampling-evaluation/view/new/experiments).

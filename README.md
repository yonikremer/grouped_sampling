# A Single Usage Is All You Need — Grouped Sampling

[![PyPI version](https://badge.fury.io/py/grouped-sampling.svg)](https://pypi.org/project/grouped-sampling/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE) [![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)

> Single-forward-pass text generation from causal LMs: 1 model call per text, not 1 per word.
>
> **Links:** [PyPI](https://pypi.org/project/grouped-sampling/) · [ISEF ROBO037](https://isef.net/project/robo037-a-single-usage-is-all-you-need) · [Eval logs (Comet)](https://www.comet.com/yonikremer/grouped-sampling-evaluation/view/new/experiments) · [Library](./src/README.md)

## Results (TED Talks / IWSLT translation, vs. naive autoregressive baseline)

| Metric | Naive baseline | Grouped sampling |
| --- | --- | --- |
| GPU hours | 33.049 ($17.87) | 0.028 ($0.015) |
| Cost / time | 1x | ~1,180x cheaper & faster |
| Quality (BERTScore) | baseline | +5–24% |

*Scope: fixed-length outputs up to group size; comparison is against the naive per-word baseline, not modern inference servers like vLLM.*

## Awards

## [FIRST PLACE in the Israeli Young Scientist and Developer Contest 2023](https://www.youngscientistsisrael.com/projects/dgymh-bqbvtsvt-shymvsh-y-yl-bmvdly-shph-sybtyym-causal-language-models)

## [Finalist at Regeneron International Science and Engineering Fair 2023](https://projectboard.world/isef/project/robo037-a-single-usage-is-all-you-need)

## Five high school credit points in data science. Grade: 99%

# Abstract:

I developed and published an open-source efficient text-generation algorithm called grouped sampling to enable affordable
and accessible AI text-generation services for everyone.

Causal language models are state-of-the-art text generation models that power many popular products like ChatGPT.
The naive text generation algorithm requires x usages of a causal language model to generate x words, 
which makes it inefficient.

Grouped sampling is an alternative algorithm, which manipulates the input text before passing it to the model,
forcing the model to predict the entire output at once. 

Grouped sampling only requires one use of a causal language model to generate text of any length, making it much more efficient.

I compared grouped sampling and the naive algorithm in translating TED talks.

The naive algorithm required 33.049 GPU hours that cost $17.87.

Grouped sampling required 0.028 GPU hours that costs $0.015.

Grouped sampling translated more accurately by 5%-24%, measured using BERT scores.

In conclusion, grouped sampling is an accurate and efficient text-generation technique. 
 
It is 1180 times faster and cheaper to run than the naive algorithm.

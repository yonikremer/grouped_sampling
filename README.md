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

## Use the package

```bash
pip install grouped-sampling
```

```python
from grouped_sampling import ReturnOnePipeLine, ReturnManyPipeLine

# one answer per prompt, output_length = max new tokens
pipe = ReturnOnePipeLine(model_name="gpt2", max_batch_size=8)
answers: list[str] = pipe.generate_batch_return_one(["Translate to French: Hello, how are you?"], output_length=32)

# several answers per prompt
many = ReturnManyPipeLine(model_name="gpt2", max_batch_size=8)
```

Any HuggingFace causal LM works as `model_name`. Sampling follows the model's `GenerationConfig` (temperature, top-p/top-k via flashinfer); greedy when sampling is off. Models load in fp16 with `device_map="auto"`.

## How it works

1. Prompts are tokenized, padded, and extended with a placeholder group of `output_length` tokens.
2. One forward pass produces a logit matrix of shape `(batch, output_length, vocab)` (`BasePipeLine.tokens_batch_to_logit_matrices`).
3. Each position is sampled independently (`LogitVectorToTokenPipeLine`, flashinfer top-p/top-k) and decoded back to text.

Trade-off: output length is fixed up front — longer than needed wastes compute, shorter than needed clips the text.

## Repo layout

```text
src/grouped_sampling/   pip-installable library (pip name: grouped-sampling)
  base_pipeline.py      model/tokenizer loading, batching, single-forward-pass logits
  return_one_pipeline.py  one completion per prompt (generate_batch_return_one)
  return_many_pipeline.py many completions per prompt
  logits_vec_to_token.py  sampling (greedy / top-p / top-k via flashinfer)
  model.py / tokenizer.py HF loading helpers
/
evaluation/             TED Talks (IWSLT) translation experiments, BERTScore scoring, Comet logs
benchmark/              throughput/latency harness (grouped sampling vs. vLLM)
tests/                  pytest suite (run_tests.py)
```

## Reproduce & verify

- Evaluation walkthrough: [`evaluation/README.md`](./evaluation/README.md); raw logs: [Comet](https://www.comet.com/yonikremer/grouped-sampling-evaluation/view/new/experiments)
- Tests: `python run_tests.py` (or `pytest tests/`)
- Benchmarks: `benchmark/throughput.py`, `benchmark/latency.py`

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

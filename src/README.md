# GROUPED SAMPLING LIBRARY

This is a library for generating text using a causal language model.

This library is using the grouped sampling algorithm.

This library supports models from huggingface hub that follow the causal language model architecture.

## Installation

```bash
# from the repo root:
pip install ./src
```

## Usage

1. Make sure You have python 3 (3.9+ is recommended), fast internet connection and the grouped sampling library.
2. Choose a Causal Language Model from huggingface hub
3. Choose a group size, which should be upper limit for the length of the generated texts.
A higher group size will cause unnecessary computations.
A lower group size will cause lower performance both in runtime and text quality.
4. `pipe = ReturnOnePipeLine(model_name=YOUR_MODEL_NAME, max_batch_size=8)`
5. `answers = pipe.generate_batch_return_one([YOUR_TEXT], output_length=YOUR_MAX_NEW_TOKENS)`
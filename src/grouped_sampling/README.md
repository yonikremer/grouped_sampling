# GROUPED SAMPLING LIBRARY

This is a library for generating text using a causal language model.

This library is using the grouped sampling algorithm.

This library supports models from huggingface hub that follow the causal language model architecture.

## Installation

```pip install grouped-sampling```

## Usage

1. Make sure You have python 3 (3.9+ is recommended), fast internet connection and the grouped sampling library.
2. Choose a Causal Language Model from huggingface hub
3. Choose a group size, which should be upper limit for the length of the generated texts.
A higher group size will cause unnecessary computations.
A lower group size will cause lower performance both in runtime and text quality.
4. `pipe = GroupedSamplingPipeLine(model_name=YOUR_MODEL_NAME, group_size=YOUR_GROUP_SIZE)`
5. `answer = pipe(YOUR_TEXT)["generated_text"]`
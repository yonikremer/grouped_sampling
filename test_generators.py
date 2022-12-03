from __future__ import annotations

from itertools import islice, product
from typing import Generator, List

import pytest
from torch import Tensor, equal

from sampling_generator import SamplingGenerator
from tree_generator import TreeGenerator
from text_generator import TextGenerator, SingleAnswer

MODEL_NAMES = "gpt2", "facebook/opt-125m"
GROUP_SIZES = 3, 1
TOP_KS = 1, 4
TOP_PS = 0.0, 0.5, 1.0
TEMPERATURES = 0.5, 1.0, 10.0
PROMPT = "Hello, world!"
repeating_prompt: str = "This is a very long text. " * 2
long_prompt: str = "This is a very long text. " * 512
EDGE_CASE_PROMPTS = [long_prompt, repeating_prompt]
TEST_PROMPT = "This is a test prompt."


def create_text_generators() -> Generator[TextGenerator, None, None]:
    top_p_sampling_gen = SamplingGenerator(model_name=MODEL_NAMES[0],
                                           group_size=GROUP_SIZES[1],
                                           top_k=None,
                                           top_p=TOP_PS[1],
                                           temp=TEMPERATURES[0])

    top_k_sampling_gen = SamplingGenerator(model_name=MODEL_NAMES[1],
                                           group_size=GROUP_SIZES[0],
                                           top_k=TOP_KS[0],
                                           top_p=None,
                                           temp=TEMPERATURES[0])

    curr_tree_gen = TreeGenerator(model_name=MODEL_NAMES[1],
                                  group_size=GROUP_SIZES[0],
                                  top_k=TOP_KS[1],
                                  top_p=TOP_PS[1],
                                  temp=TEMPERATURES[0])

    for top_k in TOP_KS:
        top_k_sampling_gen.top_k = top_k
        curr_tree_gen.top_k = top_k
        yield top_k_sampling_gen

        for top_p in TOP_PS:
            top_p_sampling_gen.top_p = top_p
            curr_tree_gen.top_p = top_p
            yield top_p_sampling_gen
            yield curr_tree_gen


def test_initialization():
    for _ in create_text_generators():
        continue


def test_str():
    for curr_text_generator in create_text_generators():
        assert isinstance(str(curr_text_generator), str), f"{str(curr_text_generator)} is not a string"


@pytest.mark.parametrize("curr_text_generator, edge_case_prompt",
                         product(islice(create_text_generators(), 3, 5), EDGE_CASE_PROMPTS))
def test_edge_cases(curr_text_generator: TextGenerator, edge_case_prompt: str):
    answer = curr_text_generator(
        prompt_s=edge_case_prompt,
        max_new_tokens=curr_text_generator.group_size * 2,
        return_full_text=True,
    )["generated_text"]
    assert isinstance(answer, str), f"{answer} is not a string. curr_generator: {str(curr_text_generator)}"
    assert len(answer) > len(edge_case_prompt), f"{answer} is not a not longer than {edge_case_prompt}"
    assert answer.startswith(edge_case_prompt[:-1]), f"{answer} doesn't start with {edge_case_prompt}"


def test_pre_and_post_process():
    """Tests the different returning options"""
    generator: TextGenerator = next(create_text_generators())
    prompt: str = "This is a test prompt"
    returned_val: SingleAnswer = generator(
        prompt_s=prompt,
        max_new_tokens=10,
        return_tensors=True,
        return_text=True,
        return_full_text=True
    )
    assert isinstance(returned_val, dict), f"{returned_val} is not a dict"
    assert "generated_text" in returned_val.keys(), f"{returned_val} doesn't contain 'generated_text'"
    assert "generated_token_ids" in returned_val.keys(), f"{returned_val} doesn't contain 'generated_token_ids'"
    returned_text = returned_val["generated_text"]
    assert isinstance(returned_text, str), f"{returned_val['generated_text']} is not a string"
    generated_token_ids: Tensor = returned_val["generated_token_ids"]
    assert isinstance(generated_token_ids,
                      Tensor), f"{returned_val['generated_token_ids']} is not a list"
    assert returned_text.startswith(
        prompt), f"{returned_text} doesn't start with {prompt}"
    prompt_tokens: Tensor
    prompt_tokens, _, _, _ = generator.preprocess(prompt)
    assert equal(generated_token_ids[:len(prompt_tokens)], prompt_tokens), \
        f"{returned_val['generated_token_ids'].tolist()} is not equal to {prompt_tokens}"
    assert isinstance(generated_token_ids[:len(prompt_tokens)], Tensor), \
        f"{returned_val['generated_token_ids']} is not a tensor"
    post_processed_text: str = generator.postprocess(
        token_ids=generated_token_ids.tolist(), num_new_tokens=10, prompt_len=len(prompt_tokens),
        return_tensors=True, return_text=True, return_full_text=True,
        clean_up_tokenization_spaces=True)['generated_text']
    assert returned_text == post_processed_text, f"{returned_text} is not equal to {post_processed_text}"

    returned_val: SingleAnswer = generator(
        prompt_s=prompt, max_new_tokens=10,
        return_tensors=True, return_text=True, return_full_text=False
    )
    returned_text = returned_val["generated_text"]
    assert not returned_text.startswith(
        prompt), f"{returned_text} doesn't start with {prompt}"
    prompt_tokens, _, _, _ = generator.preprocess(prompt)
    assert not equal(returned_val["generated_token_ids"][:len(prompt_tokens)], prompt_tokens), \
        f"{returned_val['generated_token_ids'][:len(prompt_tokens)]} is equal to {prompt_tokens}"


def test_prefix():
    """Tests that the prefix option of the methods __call__ and preprocess works"""
    generator: TextGenerator = next(create_text_generators())
    prompt: str = "test prompt"
    prefix = "This is a "
    answer: SingleAnswer = generator(
        prompt_s=prompt, max_new_tokens=10,
        return_tensors=True, return_text=True, return_full_text=True, prefix=prefix
    )
    assert prompt in answer["generated_text"], f"{answer['generated_text']} doesn't contain {prompt}"
    assert prefix not in answer["generated_text"], f"{answer['generated_text']} doesn't contain {prefix}"
    answer: str = generator(prompt_s=prompt, max_new_tokens=10,
                            return_tensors=False, return_text=True, return_full_text=False, prefix=prefix
                            )['generated_text']
    assert prompt not in answer, f"{answer} contain {prompt}"
    assert prefix not in answer, f"{answer} contain {prefix}"


def test_postfix():
    """Tests that the postfix option of the methods __call__ and preprocess works"""
    generator: TextGenerator = next(create_text_generators())
    prompt: str = "This is a"
    postfix = " Test postfix"
    answer: str = generator(
        prompt_s=prompt, max_new_tokens=10,
        return_tensors=False, return_text=True, return_full_text=True, postfix=postfix
    )["generated_text"]
    assert prompt in answer, f"{answer} doesn't contain {prompt}"
    assert postfix not in answer, f"{answer} contain {postfix}"
    answer = generator(
        prompt_s=prompt, max_new_tokens=10,
        return_tensors=False, return_text=True, return_full_text=False, postfix=postfix
    )["generated_text"]
    assert prompt not in answer, f"{answer} doesn't contain {prompt}"
    assert postfix not in answer, f"{answer} doesn't contain {postfix}"


def test_num_return_sequences():
    generator = next(create_text_generators())
    num_return_sequences = 2
    answer: List[SingleAnswer] = generator(
        prompt_s=TEST_PROMPT, max_new_tokens=8, return_tensors=False, return_text=True,
        return_full_text=True, num_return_sequences=num_return_sequences
    )
    assert len(answer) == num_return_sequences, f"len(answer) is not {num_return_sequences}"
    for curr_answer in answer:
        curr_answer: SingleAnswer
        assert curr_answer["generated_text"].startswith(TEST_PROMPT), \
            f"{curr_answer['generated_text']} doesn't start with {TEST_PROMPT}"

        assert len(curr_answer["generated_text"]) > len(TEST_PROMPT), \
            f"{curr_answer['generated_text']} is too short"


def test_max_new_tokens_is_none():
    curr_text_generator = next(create_text_generators())
    curr_text_generator.end_of_sentence_stop = True
    answer: SingleAnswer = curr_text_generator(
        prompt_s=TEST_PROMPT, max_new_tokens=None, return_tensors=False, return_text=True,
        return_full_text=True
    )
    assert answer["generated_text"].startswith(TEST_PROMPT), \
        f"{answer['generated_text']} doesn't start with {TEST_PROMPT}"

    assert len(answer["generated_text"]) > len(TEST_PROMPT), \
        f"{answer['generated_text']} is too short"


def test_call_many_prompts():
    """Tests the __call__ method when many prompts are given"""
    generator = next(create_text_generators())
    PROMPTS: List[str] = ["This is a test prompt", "This is another test prompt"]
    answers: List[SingleAnswer] = generator(
        prompt_s=PROMPTS, max_new_tokens=10,
        return_text=True, return_full_text=True
    )
    assert isinstance(answers, list), f"{answers} is not a list"
    for prompt, answer in zip(PROMPTS, answers):
        assert isinstance(answer, dict), f"{answer} is not a dict"
        assert "generated_text" in answer.keys(), f"{answer} doesn't contain the key 'generated_text'"
        assert isinstance(answer["generated_text"], str), f"{answer['generated_text']} is not a string"
        assert answer["generated_text"].startswith(prompt), f"{answer['generated_text']} doesn't start with {prompt}"


@pytest.mark.parametrize("curr_text_generator", create_text_generators())
def test_calling_generators(curr_text_generator):
    answer = curr_text_generator(
        prompt_s=PROMPT,
        max_new_tokens=curr_text_generator.group_size * 2,
    )["generated_text"]
    assert isinstance(answer, str), f"{answer} is not a string. curr_text_generator: {str(curr_text_generator)}"
    assert answer.startswith(PROMPT), f"{answer} doesn't start with {PROMPT}"
    assert len(answer) > len(PROMPT), f"{answer} is too short"


@pytest.mark.parametrize("curr_text_generator", create_text_generators())
def test_dict_conversions(curr_text_generator):
    new_generator = type(curr_text_generator).from_dict(curr_text_generator.as_dict())
    assert type(curr_text_generator) == type(new_generator)
    assert curr_text_generator.model_name == new_generator.model_name
    assert curr_text_generator.group_size == new_generator.group_size
    assert curr_text_generator.temp == new_generator.temp
    assert curr_text_generator.generation_type == new_generator.generation_type
    assert curr_text_generator.end_of_sentence_id == new_generator.end_of_sentence_id
    assert curr_text_generator.end_of_sentence_stop == new_generator.end_of_sentence_stop
    assert curr_text_generator.max_input_len == new_generator.max_input_len
    assert curr_text_generator.framework == new_generator.framework
    assert curr_text_generator.answer_length_multiplier == new_generator.answer_length_multiplier
    assert curr_text_generator.vocab_size == new_generator.vocab_size
    if hasattr(curr_text_generator, "top_k"):
        assert curr_text_generator.top_k == new_generator.top_k
    if hasattr(curr_text_generator, "top_p"):
        assert curr_text_generator.top_p == new_generator.top_p


if __name__ == '__main__':
    pytest.main([__file__])

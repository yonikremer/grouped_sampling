from __future__ import annotations

from itertools import islice, product
from typing import Generator, List

import pytest
from torch import Tensor, equal

from src.grouped_sampling import (
    GroupedGenerationPipeLine,
    GroupedSamplingPipeLine,
    GroupedTreePipeLine,
    CompletionDict,
    NoRepetitionPenalty,
)

MODEL_NAME = "gpt2"
GROUP_SIZES = 3, 1
TOP_KS = 1, 4
TOP_PS = 0.0, 0.5, 1.0
TEMPERATURES = 0.5, 1.0, 10.0
PROMPT = "Hello, world!"
repeating_prompt: str = "This is a very long text. " * 2
long_prompt: str = "This is a very long text. " * 512
EDGE_CASE_PROMPTS = [long_prompt, repeating_prompt]
TEST_PROMPT = "This is a test prompt."
TEST_PIPELINE = GroupedSamplingPipeLine(
    model_name=MODEL_NAME,
    group_size=GROUP_SIZES[0],
    top_k=TOP_KS[0],
    top_p=None,
    temp=TEMPERATURES[0],
    answer_length_multiplier=1.0,
    load_in_8bit=False,
    use_cuda=False,
)


def create_pipelines(
) -> Generator[GroupedGenerationPipeLine, None, None]:
    top_k_sampling_gen = TEST_PIPELINE
    for top_k in TOP_KS:
        top_k_sampling_gen.top_k = top_k
        yield top_k_sampling_gen

    top_p_sampling_gen = GroupedSamplingPipeLine(
        model_name=MODEL_NAME,
        group_size=GROUP_SIZES[1],
        top_k=None,
        top_p=TOP_PS[1],
        temp=TEMPERATURES[0],
        answer_length_multiplier=1.0,
        load_in_8bit=False,
        use_cuda=False,
    )
    for top_p in TOP_PS:
        top_p_sampling_gen.top_p = top_p
        yield top_p_sampling_gen

    curr_tree_gen = GroupedTreePipeLine(
        model_name=MODEL_NAME,
        group_size=GROUP_SIZES[0],
        top_k=TOP_KS[1],
        top_p=TOP_PS[1],
        temp=TEMPERATURES[1],
        answer_length_multiplier=1.0,
        repetition_penalty_strategy=NoRepetitionPenalty(),
        load_in_8bit=False,
        use_cuda=False,
    )

    for top_k in TOP_KS:
        curr_tree_gen.top_k = top_k
        for top_p in TOP_PS:
            curr_tree_gen.top_p = top_p
            yield curr_tree_gen


def test_initialization():
    for _ in create_pipelines():
        continue


def test_str():
    for curr_pipeline in create_pipelines():
        assert isinstance(str(curr_pipeline),
                          str), f"{str(curr_pipeline)} is not a string"


@pytest.mark.parametrize(
    "curr_pipeline, edge_case_prompt",
    product(islice(create_pipelines(), 3, 5), EDGE_CASE_PROMPTS),
)
def test_edge_cases(curr_pipeline: GroupedGenerationPipeLine,
                    edge_case_prompt: str):
    answer = curr_pipeline(
        prompt_s=edge_case_prompt,
        max_new_tokens=curr_pipeline.wrapped_model.group_size * 2,
        return_full_text=True,
    )["generated_text"]
    assert isinstance(
        answer, str
    ), f"{answer} is not a string. curr_pipeline: {str(curr_pipeline)}"
    assert len(answer) > len(
        edge_case_prompt
    ), f"{answer} is not a not longer than {edge_case_prompt}"
    assert answer.startswith(
        edge_case_prompt[:-1]
    ), f"{answer} doesn't start with {edge_case_prompt}"


def test_pre_and_post_process():
    """Tests the different returning options"""
    pipeline: GroupedGenerationPipeLine = TEST_PIPELINE
    prompt: str = "This is a test prompt"
    returned_val: CompletionDict = pipeline(
        prompt_s=prompt,
        max_new_tokens=10,
        return_tensors=True,
        return_text=True,
        return_full_text=True,
    )
    assert isinstance(returned_val, dict), f"{returned_val} is not a dict"
    assert ("generated_text" in returned_val.keys()
            ), f"{returned_val} doesn't contain 'generated_text'"
    assert ("generated_token_ids" in returned_val.keys()
            ), f"{returned_val} doesn't contain 'generated_token_ids'"
    returned_text = returned_val["generated_text"]
    assert isinstance(returned_text,
                      str), f"{returned_val['generated_text']} is not a string"
    generated_token_ids: Tensor = returned_val["generated_token_ids"]
    assert isinstance(
        generated_token_ids,
        Tensor), f"{returned_val['generated_token_ids']} is not a list"
    assert returned_text.startswith(
        prompt), f"{returned_text} doesn't start with {prompt}"
    prompt_tokens: Tensor
    prompt_tokens, _, _, _ = pipeline.pre_processing_strategy(prompt)
    assert equal(
        generated_token_ids[:len(prompt_tokens)], prompt_tokens
    ), f"{returned_val['generated_token_ids'].tolist()} is not equal to {prompt_tokens}"
    assert isinstance(
        generated_token_ids[:len(prompt_tokens)],
        Tensor), f"{returned_val['generated_token_ids']} is not a tensor"
    post_processed_text: str = pipeline.post_processing_strategy(
        token_ids=generated_token_ids.tolist(),
        num_new_tokens=10,
        prompt_len=len(prompt_tokens),
        return_tensors=True,
        return_text=True,
        return_full_text=True,
        clean_up_tokenization_spaces=True,
    )["generated_text"]
    assert (returned_text == post_processed_text
            ), f"{returned_text} is not equal to {post_processed_text}"

    returned_val: CompletionDict = pipeline(
        prompt_s=prompt,
        max_new_tokens=10,
        return_tensors=True,
        return_text=True,
        return_full_text=False,
    )
    returned_text = returned_val["generated_text"]
    assert not returned_text.startswith(
        prompt), f"{returned_text} doesn't start with {prompt}"
    prompt_tokens, _, _, _ = pipeline.pre_processing_strategy(prompt)
    assert not equal(
        returned_val["generated_token_ids"][:len(prompt_tokens)], prompt_tokens
    ), f"{returned_val['generated_token_ids'][:len(prompt_tokens)]} is equal to {prompt_tokens}"


def test_prefix():
    """Tests that the prefix option of the methods __call__ and preprocess works"""
    pipeline: GroupedGenerationPipeLine = TEST_PIPELINE
    prompt: str = "test prompt"
    prefix = "This is a "
    answer: CompletionDict = pipeline(
        prompt_s=prompt,
        max_new_tokens=10,
        return_tensors=True,
        return_text=True,
        return_full_text=True,
        prefix=prefix,
    )
    assert (prompt in answer["generated_text"]
            ), f"{answer['generated_text']} doesn't contain {prompt}"
    assert (prefix not in answer["generated_text"]
            ), f"{answer['generated_text']} doesn't contain {prefix}"
    answer: str = pipeline(
        prompt_s=prompt,
        max_new_tokens=10,
        return_tensors=False,
        return_text=True,
        return_full_text=False,
        prefix=prefix,
    )["generated_text"]
    assert prompt not in answer, f"{answer} contain {prompt}"
    assert prefix not in answer, f"{answer} contain {prefix}"


def test_postfix():
    """Tests that the postfix option of the methods __call__ and preprocess works"""
    pipeline: GroupedGenerationPipeLine = TEST_PIPELINE
    prompt: str = "This is a"
    postfix = " Test postfix"
    answer: str = pipeline(
        prompt_s=prompt,
        max_new_tokens=10,
        return_tensors=False,
        return_text=True,
        return_full_text=True,
        postfix=postfix,
    )["generated_text"]
    assert prompt in answer, f"{answer} doesn't contain {prompt}"
    assert postfix not in answer, f"{answer} contain {postfix}"
    answer = pipeline(
        prompt_s=prompt,
        max_new_tokens=10,
        return_tensors=False,
        return_text=True,
        return_full_text=False,
        postfix=postfix,
    )["generated_text"]
    assert prompt not in answer, f"{answer} doesn't contain {prompt}"
    assert postfix not in answer, f"{answer} doesn't contain {postfix}"


def test_max_new_tokens_is_none():
    curr_pipeline = TEST_PIPELINE
    curr_pipeline.wrapped_model.end_of_sentence_stop = True
    answer: CompletionDict = curr_pipeline(
        prompt_s=TEST_PROMPT,
        max_new_tokens=None,
        return_tensors=False,
        return_text=True,
        return_full_text=True,
    )
    assert answer["generated_text"].startswith(
        TEST_PROMPT
    ), f"{answer['generated_text']} doesn't start with {TEST_PROMPT}"

    assert len(answer["generated_text"]) > len(
        TEST_PROMPT), f"{answer['generated_text']} is too short"


def test_call_many_prompts():
    """Tests the __call__ method when many prompts are given"""
    pipeline = TEST_PIPELINE
    prompts: List[str] = [
        "This is a test prompt", "This is another test prompt"
    ]
    answers: List[CompletionDict] = pipeline(prompt_s=prompts,
                                             max_new_tokens=10,
                                             return_text=True,
                                             return_full_text=True)
    assert isinstance(answers, list), f"{answers} is not a list"
    for prompt, answer in zip(prompts, answers):
        assert isinstance(answer, dict), f"{answer} is not a dict"
        assert ("generated_text" in answer.keys()
                ), f"{answer} doesn't contain the key 'generated_text'"
        assert isinstance(answer["generated_text"],
                          str), f"{answer['generated_text']} is not a string"
        assert answer["generated_text"].startswith(
            prompt), f"{answer['generated_text']} doesn't start with {prompt}"

    pipeline.max_batch_size = 2
    prompts = prompts * 2 + [prompts[0]]
    answers: List[CompletionDict] = pipeline(
        prompt_s=prompts,
        max_new_tokens=10,
        return_tensors=True,
        return_text=True,
        return_full_text=True,
    )
    assert isinstance(answers, list), f"{answers} is not a list"
    assert len(answers) == len(
        prompts), f"{answers} doesn't have the same length as {prompts}"
    for prompt, answer in zip(prompts, answers):
        assert isinstance(answer, dict), f"{answer} is not a dict"
        assert ("generated_text" in answer.keys()
                ), f"{answer} doesn't contain the key 'generated_text'"
        assert isinstance(answer["generated_text"],
                          str), f"{answer['generated_text']} is not a string"
        assert answer["generated_text"].startswith(
            prompt), f"{answer['generated_text']} doesn't start with {prompt}"


def test_call_batch():
    pipeline = TEST_PIPELINE
    empty_batch = []
    empty_batch_answer = pipeline(empty_batch)
    assert empty_batch_answer == [], f"{empty_batch_answer} is not an empty list"
    batch_of_size_one = [TEST_PROMPT]
    batch_of_size_one_answer = pipeline(batch_of_size_one, return_full_text=True)
    assert isinstance(batch_of_size_one_answer, list), f"{batch_of_size_one_answer} is not a list"
    assert len(batch_of_size_one_answer) == 1, f"{batch_of_size_one_answer} doesn't have a length of 1"
    assert isinstance(batch_of_size_one_answer[0], dict), f"{batch_of_size_one_answer[0]} is not a dict"
    assert "generated_text" in batch_of_size_one_answer[0].keys(), \
        f"{batch_of_size_one_answer[0]} doesn't contain the key 'generated_text'"
    assert isinstance(batch_of_size_one_answer[0]["generated_text"], str), \
        f"{batch_of_size_one_answer[0]['generated_text']} is not a string"
    assert batch_of_size_one_answer[0]["generated_text"].startswith(TEST_PROMPT), \
        f"{batch_of_size_one_answer[0]['generated_text']} doesn't start with {TEST_PROMPT}"
    assert len(batch_of_size_one_answer[0]["generated_text"]) > len(TEST_PROMPT), \
        f"{batch_of_size_one_answer[0]['generated_text']} is too short"
    batch_of_size_two = [TEST_PROMPT, TEST_PROMPT]
    batch_of_size_two_answer = pipeline(batch_of_size_two, return_full_text=True)
    assert isinstance(batch_of_size_two_answer, list), f"{batch_of_size_two_answer} is not a list"
    assert len(batch_of_size_two_answer) == 2, f"{batch_of_size_two_answer} doesn't have a length of 2"
    for answer in batch_of_size_two_answer:
        assert isinstance(answer, dict), \
            f"{answer} is not a dict"
        assert "generated_text" in answer.keys(), \
            f"{answer} doesn't contain the key 'generated_text'"
        assert isinstance(answer["generated_text"], str), \
            f"{answer['generated_text']} is not a string"
        assert answer["generated_text"].startswith(TEST_PROMPT), \
            f"{answer['generated_text']} doesn't start with {TEST_PROMPT}"
        assert len(answer["generated_text"]) > len(TEST_PROMPT), \
            f"{answer['generated_text']} is too short"


def test_single_token_generation():
    pipeline = TEST_PIPELINE
    answer = pipeline(TEST_PROMPT, max_new_tokens=1, return_full_text=False)["generated_text"]
    assert isinstance(answer, str), f"{answer} is not a string"
    assert len(answer) > 0, "answer is empty"


@pytest.mark.parametrize("curr_pipeline", create_pipelines())
def test_calling_pipelines(curr_pipeline):
    answer = curr_pipeline(
        prompt_s=PROMPT,
        max_new_tokens=curr_pipeline.wrapped_model.group_size * 2,
    )["generated_text"]
    assert isinstance(
        answer, str
    ), f"{answer} is not a string. curr_pipeline: {str(curr_pipeline)}"
    assert answer.startswith(PROMPT), f"{answer} doesn't start with {PROMPT}"
    assert len(answer) > len(PROMPT), f"{answer} is too short"


@pytest.mark.parametrize("curr_pipeline", create_pipelines())
def test_dict_conversions(curr_pipeline: GroupedGenerationPipeLine):
    new_pipeline = type(curr_pipeline).from_dict(
        curr_pipeline.as_dict())
    assert type(curr_pipeline) is type(new_pipeline)
    assert curr_pipeline.model_name == new_pipeline.model_name
    assert (curr_pipeline.wrapped_model.group_size ==
            new_pipeline.wrapped_model.group_size)
    assert curr_pipeline.wrapped_model.temp == new_pipeline.wrapped_model.temp
    assert curr_pipeline.generation_type == new_pipeline.generation_type
    assert (curr_pipeline.wrapped_model.end_of_sentence_id ==
            new_pipeline.wrapped_model.end_of_sentence_id)
    assert (curr_pipeline.wrapped_model.end_of_sentence_stop ==
            new_pipeline.wrapped_model.end_of_sentence_stop)
    assert (curr_pipeline.wrapped_model.max_input_len ==
            new_pipeline.wrapped_model.max_input_len)
    assert (curr_pipeline.answer_length_multiplier ==
            new_pipeline.answer_length_multiplier)
    assert (curr_pipeline.wrapped_model.vocab_size ==
            new_pipeline.wrapped_model.vocab_size)
    if hasattr(curr_pipeline, "top_k"):
        assert curr_pipeline.top_k == new_pipeline.top_k
    if hasattr(curr_pipeline, "top_p"):
        assert curr_pipeline.top_p == new_pipeline.top_p


if __name__ == "__main__":
    pytest.main([__file__])

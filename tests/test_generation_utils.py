"""Tests for the generation_utils.py file in the grouped_sampling module of the project"""
import random
from typing import List

import numpy as np
import pytest
import torch
from torch import all, isclose, eq, tensor, cat
from transformers import AutoTokenizer

from src.grouped_sampling import GroupedGenerationUtils, TokenIDS, DEFAULT_REPETITION_PENALTY
from src.grouped_sampling.generation_utils import safe_cat_batch
from src.grouped_sampling.model import get_model
from src.grouped_sampling.tokenizer import get_padding_id, get_end_of_text_id


def create_wrapped_models() -> List[GroupedGenerationUtils]:
    """Creates a list of wrapped models to test"""
    model = get_model(
        model_name="gpt2",
    )
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    config = model.config
    wrapped_model_example1 = GroupedGenerationUtils(
        model=model,
        vocab_size=50257,
        max_input_len=1024,
        end_of_sentence_stop=False,
        repetition_penalty_strategy=DEFAULT_REPETITION_PENALTY,
        end_of_sentence_id=get_end_of_text_id(tokenizer, config),
        padding_id=get_padding_id(tokenizer),
        load_in_8bit=False,
    )
    return [wrapped_model_example1]


@pytest.mark.parametrize("wrapped_model", create_wrapped_models())
def test_get_prob_mat_batch(wrapped_model: GroupedGenerationUtils):
    """Tests that the batched version of get_prob_mat is the same as the non-batched version"""
    single_sequence: TokenIDS = [1, 2, 3, 4, 5]
    batch_of_size_two: List[TokenIDS] = [single_sequence, single_sequence]
    un_batched_prob_mat = wrapped_model.get_prob_mat(single_sequence, 0, 10)
    assert un_batched_prob_mat.shape == (10, wrapped_model.vocab_size)
    prob_mat_batch = wrapped_model.get_prob_mat_batch(batch_of_size_two, [0, 0], 10)
    assert prob_mat_batch.shape == (len(batch_of_size_two), 10, wrapped_model.vocab_size)
    batch_of_size_two: List[TokenIDS] = [single_sequence, single_sequence]
    prob_mats = wrapped_model.get_prob_mat_batch(batch_of_size_two, [0, 0], 10)
    assert prob_mats.shape == (len(batch_of_size_two), 10, wrapped_model.vocab_size)
    assert prob_mats[0].shape == prob_mats[0].shape == (10, wrapped_model.vocab_size)
    assert all(isclose(prob_mats[0], prob_mats[1])), (prob_mats[0], prob_mats[1])
    for i in range(prob_mat_batch.shape[1]):
        assert all(isclose(prob_mat_batch[0, i], prob_mat_batch[1, i], atol=1e-4, rtol=1e-4)),\
            (i, prob_mat_batch[0, i], prob_mat_batch[1, i])


@pytest.mark.parametrize("wrapped_model", create_wrapped_models())
def test_get_logit_mat(wrapped_model: GroupedGenerationUtils):
    single_sequence: TokenIDS = [1, 2, 3, 4, 5]
    batch_of_size_two: List[TokenIDS] = [single_sequence, single_sequence]
    un_batched_logits_mat = wrapped_model.get_logits_matrix(single_sequence, 10)
    assert un_batched_logits_mat.shape == (10, wrapped_model.vocab_size)
    logits_mat_batch = wrapped_model.get_logit_mat_batch(batch_of_size_two, 10)
    assert logits_mat_batch.shape == (len(batch_of_size_two), 10, wrapped_model.vocab_size)
    for i in range(logits_mat_batch.shape[1]):
        assert all(isclose(logits_mat_batch[0, i], logits_mat_batch[1, i], atol=1e-4, rtol=1e-4)),\
            (i, logits_mat_batch[0, i], logits_mat_batch[1, i])

    for i in range(logits_mat_batch.shape[1]):
        assert all(isclose(logits_mat_batch[0, i], un_batched_logits_mat[i], atol=1e-4, rtol=1e-4)),\
            (i, logits_mat_batch[0, i], un_batched_logits_mat[i])


@pytest.mark.parametrize("wrapped_model", create_wrapped_models())
def test_prepare_model_kwargs_batch(wrapped_model: GroupedGenerationUtils):
    """Tests that the batched version of prepare_model_kwargs is the same as the non-batched version"""
    tokens: TokenIDS = [1, 2, 3, 4, 5]
    output_size = 15
    model_kwargs_non_batch = wrapped_model.prepare_model_kwargs(tokens, output_size)
    print("model_kwargs_non_batch: ", model_kwargs_non_batch)
    non_batch_ids = model_kwargs_non_batch["input_ids"]
    print("non_batch_ids: ", model_kwargs_non_batch["input_ids"])
    print("non_batch_ids: ", non_batch_ids)
    model_kwargs_batch = wrapped_model.prepare_model_kwargs_batch([tokens, tokens], output_size)
    assert model_kwargs_non_batch.keys() == model_kwargs_batch.keys()

    batch_ids = model_kwargs_batch["input_ids"]
    assert (1, len(tokens) + output_size - 1) == non_batch_ids.shape, non_batch_ids
    assert (2, len(tokens) + output_size - 1) == batch_ids.shape
    assert all(eq(non_batch_ids, batch_ids))
    for key in model_kwargs_non_batch.keys():
        if key in ["output_attentions", "output_hidden_states"]:
            assert model_kwargs_non_batch[key] == model_kwargs_batch[key]
        else:
            assert all(eq(model_kwargs_batch[key], model_kwargs_non_batch[key])), f"key {key} is not equal"
    actual_batch = [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
    max_len = max(len(seq) for seq in actual_batch)
    model_kwargs = wrapped_model.prepare_model_kwargs_batch(actual_batch, output_size)
    input_ids = model_kwargs["input_ids"]
    assert (len(actual_batch), max_len + output_size - 1) == input_ids.shape


def test_safe_cat_batch():
    """Tests that safe_cat_batch works as expected on a few examples"""
    # check that it raises ValueError with empty lists
    with pytest.raises(ValueError):
        safe_cat_batch([])
    # check that it works with a single tensor
    example_tensor = tensor([1, 2, 3])
    catted = safe_cat_batch([example_tensor])
    assert all(eq(catted, example_tensor))
    # check that it works with multiple tensors
    example_tensor_2 = tensor([4, 5, 6])
    catted = safe_cat_batch([example_tensor, example_tensor_2])
    assert all(eq(catted, cat([example_tensor, example_tensor_2])))


@pytest.mark.parametrize("wrapped_model", create_wrapped_models())
def test_str_repr(wrapped_model: GroupedGenerationUtils):
    """Tests that the str and repr methods work as expected"""
    repr_ans = repr(wrapped_model)
    str_ans = str(wrapped_model)
    assert str_ans == repr_ans
    assert "GroupedGenerationUtils" in str_ans
    assert "temp" in str_ans
    assert "repetition_penalty" in str_ans
    assert "end_of_sentence_stop" in str_ans
    assert isinstance(str_ans, str)


if __name__ == '__main__':
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    torch.set_float32_matmul_precision('high')
    pytest.main()

"""Tests for the generation_utils.py file in the grouped_sampling module of the project"""
from typing import List

import pytest
from torch import all, isclose, eq, tensor, cat

from src.grouped_sampling import GroupedGenerationUtils, TokenIDS, DEFAULT_REPETITION_PENALTY
from src.grouped_sampling.generation_utils import safe_cat_batch


def create_wrapped_models() -> List[GroupedGenerationUtils]:
    """Creates a list of wrapped models to test"""
    wrapped_model_example1 = GroupedGenerationUtils(
        model_name="gpt2",
        group_size=8,
        vocab_size=50257,
        max_input_len=1024,
        end_of_sentence_stop=False,
        repetition_penalty_strategy=DEFAULT_REPETITION_PENALTY,
        end_of_sentence_id=50256,
        padding_id=0,
    )
    return [wrapped_model_example1]


@pytest.mark.parametrize("wrapped_model", create_wrapped_models())
def test_get_prob_mat_batch(wrapped_model: GroupedGenerationUtils):
    """Tests that the batched version of get_prob_mat is the same as the non-batched version"""
    single_sequence: TokenIDS = [1, 2, 3, 4, 5]
    batch_of_size_two: List[TokenIDS] = [single_sequence, single_sequence]
    un_batched_prob_mat = wrapped_model.get_prob_mat(single_sequence, 0)
    assert un_batched_prob_mat.shape == (wrapped_model.group_size, wrapped_model.vocab_size)
    prob_mat_batch = wrapped_model.get_prob_mat_batch(batch_of_size_two, [0, 0])
    assert prob_mat_batch.shape == (len(batch_of_size_two), wrapped_model.group_size, wrapped_model.vocab_size)
    assert all(isclose(prob_mat_batch[0, :, :], un_batched_prob_mat))
    batch_of_size_two: List[TokenIDS] = [single_sequence, single_sequence]
    prob_mats = wrapped_model.get_prob_mat_batch(batch_of_size_two, [0, 0])
    assert all(isclose(prob_mats[0], prob_mats[1]))


@pytest.mark.parametrize("wrapped_model", create_wrapped_models())
def test_prepare_model_kwargs_batch(wrapped_model: GroupedGenerationUtils):
    """Tests that the batched version of prepare_model_kwargs is the same as the non-batched version"""
    tokens: TokenIDS = [1, 2, 3, 4, 5]
    model_kwargs_non_batch = wrapped_model.prepare_model_kwargs(tokens)
    model_kwargs_batch = wrapped_model.prepare_model_kwargs_batch([tokens] * 2)
    assert model_kwargs_non_batch.keys() == model_kwargs_batch.keys()
    non_batch_ids = model_kwargs_non_batch["input_ids"]
    batch_ids = model_kwargs_batch["input_ids"]
    assert (1, len(tokens) + wrapped_model.group_size - 1) == non_batch_ids.shape
    assert (2, len(tokens) + wrapped_model.group_size - 1) == batch_ids.shape
    assert all(eq(non_batch_ids, batch_ids))
    for key in model_kwargs_non_batch.keys():
        assert all(eq(model_kwargs_batch[key], model_kwargs_non_batch[key])), f"key {key} is not equal"
    actual_batch = [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
    max_len = max(len(seq) for seq in actual_batch)
    model_kwargs = wrapped_model.prepare_model_kwargs_batch(actual_batch)
    input_ids = model_kwargs["input_ids"]
    assert (len(actual_batch), max_len + wrapped_model.group_size - 1) == input_ids.shape


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
    assert "group_size" in str_ans
    assert "end_of_sentence_stop" in str_ans
    assert isinstance(str_ans, str)


if __name__ == '__main__':
    pytest.main()

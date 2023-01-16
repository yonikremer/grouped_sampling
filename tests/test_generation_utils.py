from typing import List

import pytest
import torch

from src.grouped_sampling import GroupedGenerationUtils, TokenIDS, DEFAULT_REPETITION_PENALTY


def create_wrapped_models() -> List[GroupedGenerationUtils]:
    return [GroupedGenerationUtils(
        model_name="gpt2",
        group_size=8,
        vocab_size=50257,
        max_input_len=1024,
        end_of_sentence_stop=False,
        repetition_penalty_strategy=DEFAULT_REPETITION_PENALTY,
        end_of_sentence_id=50256,
        padding_id=0,
    )]


@pytest.mark.parametrize("wrapped_model", create_wrapped_models())
def test_get_prob_mat_batch(wrapped_model: GroupedGenerationUtils):
    single_sequence: TokenIDS = [1, 2, 3, 4, 5]
    batch_of_size_one: List[TokenIDS] = [single_sequence]
    un_batched_prob_mat = wrapped_model.get_prob_mat(single_sequence, 0)
    assert un_batched_prob_mat.shape == (wrapped_model.group_size, wrapped_model.vocab_size)
    prob_mat_batch = wrapped_model.get_prob_mat_batch(batch_of_size_one, [0])
    assert prob_mat_batch.shape == (len(batch_of_size_one), wrapped_model.group_size, wrapped_model.vocab_size)
    assert torch.all(torch.isclose(prob_mat_batch[0], un_batched_prob_mat))
    batch_of_size_two: List[TokenIDS] = [single_sequence, single_sequence]
    prom_mats = wrapped_model.get_prob_mat_batch(batch_of_size_two)
    assert torch.all(torch.isclose(prom_mats[0], prom_mats[1]))


@pytest.mark.parametrize("wrapped_model", create_wrapped_models())
def test_prepare_model_kwargs_batch(wrapped_model: GroupedGenerationUtils):
    tokens: TokenIDS = [1, 2, 3, 4, 5]
    model_kwargs_non_batch = wrapped_model.prepare_model_kwargs(tokens)
    model_kwargs_batch = wrapped_model.prepare_model_kwargs_batch([tokens])
    assert model_kwargs_non_batch.keys() == model_kwargs_batch.keys()
    non_batch_ids = model_kwargs_non_batch["input_ids"]
    batch_ids = model_kwargs_batch["input_ids"]
    assert (1, len(tokens) + wrapped_model.group_size - 1) == non_batch_ids.shape == batch_ids.shape
    assert torch.all(torch.eq(non_batch_ids, batch_ids))
    for key in model_kwargs_non_batch.keys():
        assert torch.all(torch.eq(model_kwargs_batch[key], model_kwargs_non_batch[key])), f"key {key} is not equal"

    actual_batch = [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
    wrapped_model.prepare_model_kwargs_batch(actual_batch)

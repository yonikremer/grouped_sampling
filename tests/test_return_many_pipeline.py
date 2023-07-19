import pytest

# Generated by CodiumAI

import torch

from fix_bitsandbytes import fix_ld_library_path
from src.grouped_sampling.return_many_pipeline import ReturnManyPipeLine

"""
Code Analysis

Main functionalities:
The ReturnManyPipeLine class is a subclass of BasePipeLine that generates multiple output sequences for each input prompt. It uses a probability processor to modify the probabilities of the model's output and then samples from the modified probabilities to generate multiple output sequences. The class provides two main functionalities: logits_to_tokens_return_many, which converts a batch of logit matrices to tokens, and generate_return_many, which generates multiple output sequences for a batch of input prompts.

Methods:
- logits_to_tokens_return_many: Converts a batch of logit matrices to tokens and returns a Tensor of shape (batch_size, output_seq_len, num_return_sequences) with num_return_sequences generated output sequences for each prompt in the tokens.
- generate_return_many: Generates multiple output sequences for a batch of input prompts. It first tokenizes and pads the input prompts, then converts the tokens to logit matrices using tokens_batch_to_logit_matrices, and finally generates multiple output sequences using logits_to_tokens_return_many.

Fields:
- max_batch_size: The maximum batch size to use.
- tokenizer: The tokenizer used to tokenize the input prompts.
- model: The model used to generate the output sequences.
- device: The device on which the model is run.
- max_total_len: The maximum length of the input and output sequences.
- temperature: The temperature used to modify the probabilities of the model's output.
- probability_processor: The probability processor used to modify the probabilities of the model's output.
"""


class TestLogitsToTokensReturnMany:
    @staticmethod
    def setup_method():
        fix_ld_library_path()

    pipeline = ReturnManyPipeLine("gpt2")

    #  Tests with valid input for logits and num_return_sequences
    @pytest.mark.parametrize(
        "batch_size, output_length, vocab_size, num_return_sequences",
        [(1, 1, 1, 1), (2, 3, 4, 5), (10, 20, 30, 40), (100, 200, 300, 400)],
    )
    def test_valid_input(
        self, batch_size, output_length, vocab_size, num_return_sequences
    ):
        logits = torch.randn(batch_size, output_length, vocab_size)
        output = self.pipeline.logits_to_tokens_return_many(
            logits, num_return_sequences
        )
        assert output.shape == (batch_size, num_return_sequences, output_length)
        assert (output >= 0).all() and (output <= vocab_size - 1).all()

    #  Tests with top_p=0.5 and top_k=10
    def test_top_p_and_top_k(self):
        pipeline = ReturnManyPipeLine("gpt2", top_p=0.5, top_k=10)
        vocab_size = 20
        batch_size = 2
        output_length = 3
        logits = torch.randn(batch_size, output_length, vocab_size)
        num_return_sequences = 2
        output = pipeline.logits_to_tokens_return_many(logits, num_return_sequences)
        assert output.shape == (batch_size, num_return_sequences, output_length)
        assert (output >= 0).all() and (output <= vocab_size - 1).all()

    #  Tests with various edge cases
    def test_edge_cases(self):
        pipeline = ReturnManyPipeLine(
            "gpt2", top_p=0.01, top_k=0, minimum_tokens_to_keep=1, temperature=0.1
        )
        logits = torch.randn(1, 1, 1)
        num_return_sequences = 1
        output = pipeline.logits_to_tokens_return_many(logits, num_return_sequences)
        assert output.shape == (1, 1, 1)
        assert (output == 0).all()

    #  Tests that generate_return_many returns a single sequence for a single prompt with valid inputs
    def test_generate_return_many_single_prompt_single_sequence(self):
        prompt = "Hello, how are you?"
        output_length = 10
        num_return_sequences = 1
        result = self.pipeline.generate_return_many(
            [prompt], output_length, num_return_sequences
        )
        assert len(result) == 1
        assert len(result[0]) == 1

    #  Tests that generate_return_many returns multiple sequences for a single prompt with valid inputs
    def test_generate_return_many_single_prompt_multiple_sequences(self):
        prompt = "Hello, how are you?"
        output_length = 10
        num_return_sequences = 3
        result = self.pipeline.generate_return_many(
            [prompt], output_length, num_return_sequences
        )
        assert len(result) == 1
        assert len(result[0]) == 3

    #  Tests that generate_return_many returns a single sequence for multiple prompts with valid inputs
    def test_generate_return_many_multiple_prompts_single_sequence(self):
        prompts = ["Hello, how are you?", "What is your name?"]
        output_length = 10
        num_return_sequences = 1
        result = self.pipeline.generate_return_many(
            prompts, output_length, num_return_sequences
        )
        assert len(result) == 2
        assert len(result[0]) == 1
        assert len(result[1]) == 1

    #  Tests that generate_return_many returns multiple sequences for multiple prompts with valid inputs
    def test_generate_return_many_multiple_prompts_multiple_sequences(self):
        prompts = ["Hello, how are you?", "What is your name?"]
        output_length = 10
        num_return_sequences = 3
        result = self.pipeline.generate_return_many(
            prompts, output_length, num_return_sequences
        )
        assert len(result) == 2
        assert len(result[0]) == 3
        assert len(result[1]) == 3

    #  Tests that generate_return_many raises a ValueError when output_length is invalid
    def test_generate_return_many_invalid_output_length(self):
        prompts = ["Hello, how are you?", "What is your name?"]
        output_length = -1
        num_return_sequences = 3
        with pytest.raises(ValueError):
            self.pipeline.generate_return_many(
                prompts, output_length, num_return_sequences
            )

    #  Tests that generate_return_many raises a ValueError when num_return_sequences is invalid
    def test_generate_return_many_invalid_num_return_sequences(self):
        prompts = ["Hello, how are you?", "What is your name?"]
        output_length = 10
        num_return_sequences = -1
        with pytest.raises(ValueError):
            self.pipeline.generate_return_many(
                prompts, output_length, num_return_sequences
            )

    # Tests pipeline with 8bit quantization
    def test_quantization(self):
        pipeline = ReturnManyPipeLine(
            "fxmarty/tiny-llama-fast-tokenizer",
            model_kwargs={"load_in_8bit": True}
        )
        prompt = "Hello, how are you?"
        output_length = 10
        num_return_sequences = 3
        pipeline.generate_return_many(
            [prompt], output_length, num_return_sequences
        )

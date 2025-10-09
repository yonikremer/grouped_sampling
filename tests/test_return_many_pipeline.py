import pytest
import torch
from transformers import GenerationConfig

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
    generation_config = GenerationConfig.from_pretrained("gpt2")
    generation_config.top_p = 0.9
    generation_config.top_k = 50
    generation_config.do_sample = True
    pipeline = ReturnManyPipeLine("gpt2",
                                  generation_config=generation_config,
                                  max_batch_size=2,
                                  seed=42)

    #  Tests that generate_return_many returns a single sequence for a single prompt with valid inputs
    def test_generate_return_many_single_prompt_single_sequence(self):
        prompt = "Hello, how are you?"
        output_length = 10
        num_return_sequences = 1
        result = self.pipeline.generate_return_many([prompt], output_length,
                                                    num_return_sequences)
        assert len(result) == 1
        assert len(result[0]) == 1
        assert isinstance(result[0][0], str)

    #  Tests that generate_return_many returns multiple sequences for a single prompt with valid inputs
    def test_generate_return_many_single_prompt_multiple_sequences(self):
        prompt = "Hello, how are you?"
        output_length = 10
        num_return_sequences = 3
        result = self.pipeline.generate_return_many([prompt], output_length,
                                                    num_return_sequences)
        assert len(result) == 1
        assert len(result[0]) == 3
        assert all(isinstance(token, str) for token in result[0])

    #  Tests that generate_return_many returns a single sequence for multiple prompts with valid inputs
    def test_generate_return_many_multiple_prompts_single_sequence(self):
        prompts = ["Hello, how are you?", "What is your name?"]
        output_length = 10
        num_return_sequences = 1
        result = self.pipeline.generate_return_many(prompts, output_length,
                                                    num_return_sequences)
        assert len(result) == 2
        assert len(result[0]) == 1
        assert len(result[1]) == 1
        assert all(isinstance(response, str) for r in result for response in r)

    #  Tests that generate_return_many returns multiple sequences for multiple prompts with valid inputs
    def test_generate_return_many_multiple_prompts_multiple_sequences(self):
        prompts = ["Hello, how are you?", "What is your name?"]
        output_length = 10
        num_return_sequences = 3
        result = self.pipeline.generate_return_many(prompts, output_length,
                                                    num_return_sequences)
        assert len(result) == 2
        assert len(result[0]) == 3
        assert len(result[1]) == 3

    #  Tests that generate_return_many raises a ValueError when output_length is invalid
    def test_generate_return_many_invalid_output_length(self):
        prompts = ["Hello, how are you?", "What is your name?"]
        output_length = -1
        num_return_sequences = 3
        with pytest.raises(ValueError):
            self.pipeline.generate_return_many(prompts, output_length,
                                               num_return_sequences)

    #  Tests that generate_return_many raises a ValueError when num_return_sequences is invalid
    def test_generate_return_many_invalid_num_return_sequences(self):
        prompts = ["Hello, how are you?", "What is your name?"]
        output_length = 10
        num_return_sequences = -1
        with pytest.raises(ValueError):
            self.pipeline.generate_return_many(prompts, output_length,
                                               num_return_sequences)

    # Tests the pipeline with 8bit quantization
    def test_quantization(self):
        generation_config = GenerationConfig.from_pretrained("gpt2")
        generation_config.top_k = 50
        generation_config.do_sample = True
        pipeline = ReturnManyPipeLine(
            "fxmarty/tiny-llama-fast-tokenizer",
            model_kwargs={"load_in_8bit": True},
            generation_config=generation_config,
            max_batch_size=2,
            seed=42,
        )
        prompt = "Hello, how are you?"
        output_length = 10
        num_return_sequences = 3
        res = pipeline.generate_return_many([prompt], output_length,
                                            num_return_sequences)
        assert len(res) == 1
        assert len(res[0]) == 3
        assert all(isinstance(token, str) for token in res[0])

    # Tests for generate_return_many_tokens
    def test_generate_return_many_tokens_shape(self):
        prompt = "Hello, how are you?"
        output_length = 5
        num_return_sequences = 2
        # Tokenize and pad prompt to get tensor input
        padded_tokens = self.pipeline.tokenize_and_pad([prompt], output_length)
        result = self.pipeline.generate_return_many_tokens(
            padded_tokens, output_length, num_return_sequences)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, num_return_sequences, output_length)

    def test_generate_return_many_tokens_multiple_prompts(self):
        prompts = ["Hello, how are you?", "What is your name?"]
        output_length = 4
        num_return_sequences = 3
        padded_tokens = self.pipeline.tokenize_and_pad(prompts, output_length)
        result = self.pipeline.generate_return_many_tokens(
            padded_tokens, output_length, num_return_sequences)
        assert result.shape == (2, num_return_sequences, output_length)
        assert isinstance(result, torch.Tensor)

    def test_generate_return_many_tokens_invalid_output_length(self):
        prompt = "Hello, how are you?"
        output_length = 0
        num_return_sequences = 2
        padded_tokens = self.pipeline.tokenize_and_pad([prompt], 1)
        with pytest.raises(ValueError):
            self.pipeline.generate_return_many_tokens(padded_tokens,
                                                      output_length,
                                                      num_return_sequences)

    def test_generate_return_many_tokens_invalid_num_return_sequences(self):
        prompt = "Hello, how are you?"
        output_length = 5
        num_return_sequences = 0
        padded_tokens = self.pipeline.tokenize_and_pad([prompt], output_length)
        with pytest.raises(ValueError):
            self.pipeline.generate_return_many_tokens(padded_tokens,
                                                      output_length,
                                                      num_return_sequences)

    def test_generate_return_many_tokens_invalid_prompt_shape(self):
        prompt = "Hello, how are you?"
        output_length = 5
        num_return_sequences = 2
        # Create a 1D tensor instead of 2D
        padded_tokens = self.pipeline.tokenize_and_pad([prompt],
                                                       output_length).view(-1)
        with pytest.raises(ValueError):
            self.pipeline.generate_return_many_tokens(padded_tokens,
                                                      output_length,
                                                      num_return_sequences)

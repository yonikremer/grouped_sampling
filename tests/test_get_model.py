# Generated by CodiumAI
import os
import random
from typing import Callable

import numpy as np
import pytest
import torch
from huggingface_hub.utils import RepositoryNotFoundError
from torch import inference_mode, cuda

# noinspection PyProtectedMember
from torch._dynamo import OptimizedModule
from torch.nn import Module
from transformers import AutoTokenizer, PretrainedConfig

from fix_bitsandbytes import fix_ld_library_path
from src.grouped_sampling.model import get_model

"""
Code Analysis

Objective:
The objective of the 'get_model' function is to load a pre-trained model from the Hugging Face Transformers library, with the option to use 8-bit quantization and CUDA acceleration, and return the compiled model.

Inputs:
- model_name (str): the name or path of the pre-trained model to load
- load_in_8bit (bool): whether to use 8-bit quantization for the model (default False)
- **kwargs: additional keyword arguments to pass to the 'from_pretrained' method of the 'AutoModelForCausalLM' class

Flow:
1. Determine if 8-bit quantization and CUDA acceleration should be used based on the input parameters.
2. Create a dictionary of keyword arguments to pass to the 'from_pretrained' method of the 'AutoModelForCausalLM' class, including the model name/path, 8-bit quantization flag, resume download flag, and any additional keyword arguments passed in.
3. If 8-bit quantization is being used, set the 'device_map' key in the dictionary to 'auto'.
4. Load the pre-trained model using the 'from_pretrained' method of the 'AutoModelForCausalLM' class with the created dictionary of keyword arguments.
5. If CUDA acceleration is being used and available, move the model to the GPU.
6. If the model has an 'eval' method, call it.
7. Compile the model using the 'compile' method of the 'torch' module.
8. Return the compiled model.

Outputs:
- model (PreTrainedModel): the loaded and compiled pre-trained model

Additional aspects:
- If the model is too large for the GPU and CUDA acceleration is being used, a warning is issued and the model is moved to the CPU.
- If CUDA acceleration is not available, a warning is issued and the model is loaded on the CPU.
"""


class TestGetModel:
    @staticmethod
    def setup_method():
        fix_ld_library_path()

    @staticmethod
    def validate_model(model):
        assert isinstance(
            model, OptimizedModule
        ), "The model is not an instance of OptimizedModule"
        assert isinstance(
            model, Module
        ), "The model is not an instance of torch.nn.Module"
        assert hasattr(model, "config"), "The model does not have a config attribute"
        assert isinstance(
            model.config, PretrainedConfig
        ), "The model config is not an instance of PretrainedConfig"
        if hasattr(model.config, "use_cache"):
            assert not model.config.use_cache, "The model config use_cache is not False"
        else:
            print("The model config does not have a use_cache attribute")
        assert hasattr(model, "device"), "The model does not have a device attribute"
        assert isinstance(
            model.device, torch.device
        ), "The model device is not an instance of torch.device"
        if cuda.is_available():
            assert (
                model.device.type == "cuda"
            ), "CUDA is avilable and the model device is not utilizing it"
        else:
            assert (
                model.device.type == "cpu"
            ), "CUDA is not avilable and the model device is not on the CPU"
        assert isinstance(model, Callable), "The model is not callable"

    #  Tests that the function returns a PreTrainedModel object
    def test_returns_model_object(self):
        model = get_model("gpt2")
        self.validate_model(model)

    #  Tests that the function loads the model with the correct kwargs
    def test_loads_correct_kwargs(self):
        model_name = "gpt2"
        kwargs = {"output_hidden_states": True}
        model = get_model(model_name, **kwargs)
        assert model.config.output_hidden_states == kwargs["output_hidden_states"]
        self.validate_model(model)

    #  Tests that the function loads the model in 8bit correctly
    def test_loads_model_in_8bit(self):
        model_name = "gpt2"
        model = get_model(model_name, load_in_8bit=True)
        self.validate_model(model)

    @inference_mode()
    # Tests that the model gives the same output when getting the same input
    def test_model_gives_same_output(self):
        model_name = "gpt2"
        model = get_model(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        input_ids = tokenizer("Hello, my dog is cute", return_tensors="pt")["input_ids"]
        input_ids = input_ids.cuda()
        output1 = model(input_ids)
        logits1 = output1.logits
        output2 = model(input_ids)
        logits2 = output2.logits
        assert torch.equal(logits1, logits2)

    @inference_mode()
    # Tests that the model gives the same output whith and without batching
    def test_model_gives_same_output_with_and_without_batching(self):
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        model_name = "fxmarty/tiny-llama-fast-tokenizer"
        model = get_model(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        prompts = ["Hello, my dog is cute"] * 2
        input_ids_batch = tokenizer(prompts, return_tensors="pt")["input_ids"].cuda()
        with torch.no_grad():
            output_batch = model(input_ids_batch)
        logits_batch = output_batch.logits
        input_ids = tokenizer(prompts[0], return_tensors="pt")["input_ids"].cuda()
        with torch.no_grad():
            output = model(input_ids)
        logits = output.logits
        assert torch.equal(logits_batch[0], logits_batch[1])
        assert logits_batch.shape == (2, input_ids.shape[1], model.config.vocab_size)
        assert logits.shape == (1, input_ids.shape[1], model.config.vocab_size)
        assert logits.dtype == logits_batch.dtype
        for i in range(input_ids.shape[1]):
            assert torch.isclose(
                logits_batch[0][i], logits[0][i], atol=1e-5, rtol=1e-4
            ).all(), (
                f"Index {i} does not match. got {logits_batch[0][i]} and {logits[0][i]}"
            )

    def test_nonexistent_model(self):
        with pytest.raises(RepositoryNotFoundError):
            get_model("nonexistent-model")

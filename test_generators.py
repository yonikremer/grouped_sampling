from itertools import islice
from unittest import TestCase, main
from typing import Generator, Union, Dict, List

from torch import Tensor, equal

from sampling_generator import SamplingGenerator
from tree_generator import TreeGenerator
from text_generator import TextGenerator


class TestTextGenerator(TestCase):
    """Testing the classes that inherit from TextGenerator."""

    MODEL_NAMES = "gpt2", "facebook/opt-125m"
    GROUP_SIZES = 3, 1
    TOP_KS = 1, 4
    TOP_PS = 0.0, 0.5, 1.0
    TEMPERATURES = 0.5, 1.0, 10.0
    PROMPT = "Hello, world!"

    def create_text_generators(self) -> Generator[TextGenerator, None, None]:
        curr_tree_gen = TreeGenerator(model_name=self.MODEL_NAMES[0],
                                      group_size=self.GROUP_SIZES[0],
                                      top_k=self.TOP_KS[0],
                                      top_p=self.TOP_PS[0],
                                      temp=self.TEMPERATURES[0],
                                      end_of_sentence_stop=True)
        top_p_sampling_gen = SamplingGenerator(model_name=self.MODEL_NAMES[0],
                                               group_size=self.GROUP_SIZES[0],
                                               top_k=None,
                                               top_p=self.TOP_PS[0],
                                               temp=self.TEMPERATURES[0],
                                               end_of_sentence_stop=True)
        yield top_p_sampling_gen
        yield curr_tree_gen

        for model_name in self.MODEL_NAMES:
            if self.MODEL_NAMES.index(model_name) > 0:
                del top_p_sampling_gen, top_k_sampling_gen, curr_tree_gen
            top_p_sampling_gen = SamplingGenerator(model_name=model_name,
                                                   group_size=self.GROUP_SIZES[0],
                                                   top_k=None,
                                                   top_p=self.TOP_PS[0],
                                                   temp=self.TEMPERATURES[0])

            top_k_sampling_gen = SamplingGenerator(model_name=model_name,
                                                   group_size=self.GROUP_SIZES[0],
                                                   top_k=self.TOP_KS[0],
                                                   top_p=None,
                                                   temp=self.TEMPERATURES[0])

            curr_tree_gen = TreeGenerator(model_name=model_name,
                                          group_size=self.GROUP_SIZES[0],
                                          top_k=self.TOP_KS[0],
                                          top_p=self.TOP_PS[0],
                                          temp=self.TEMPERATURES[0])

            for group_size in self.GROUP_SIZES:
                top_p_sampling_gen.group_size = group_size
                top_k_sampling_gen.group_size = group_size
                curr_tree_gen.group_size = group_size

                for top_k in self.TOP_KS:
                    top_k_sampling_gen.top_k = top_k
                    curr_tree_gen.top_k = top_k

                    for top_p in self.TOP_PS:
                        top_p_sampling_gen.top_p = top_p
                        curr_tree_gen.top_p = top_p

                        for temp in self.TEMPERATURES:
                            top_p_sampling_gen.temp = temp
                            top_k_sampling_gen.temp = temp
                            curr_tree_gen.temp = temp

                            yield top_p_sampling_gen
                            yield top_k_sampling_gen
                            yield curr_tree_gen

    def test_calling_generators(self):
        for curr_text_generator in self.create_text_generators():
            with self.subTest(curr_text_generator=str(curr_text_generator)) as sub:
                print(f"started sub test: {sub}")
                answer = curr_text_generator(
                    prompt_s=self.PROMPT,
                    max_new_tokens=curr_text_generator.group_size * 2,
                )["generated_text"]
                self.assertIsInstance(answer, str, f"{answer} is not a string. "
                                                   f"curr_text_generator: {str(curr_text_generator)}")
                self.assertGreater(len(answer), len(self.PROMPT), f"{answer} is not a not longer than {self.PROMPT}. "
                                                                  f"curr_text_generator: {str(curr_text_generator)}")
                self.assertTrue(answer.startswith(self.PROMPT), f"{answer} doesn't start with {self.PROMPT}. "
                                                                f"curr_text_generator: {str(curr_text_generator)}")
                print(f"ended sub test: {sub}")

    repeating_prompt: str = "This is a very long text. " * 2
    long_prompt: str = "This is a very long text. " * 2048
    edge_cases = [long_prompt, repeating_prompt]

    def test_edge_cases(self):
        """Tests the generators with long inputs. (an edge case)"""

        for prompt in self.edge_cases:
            for curr_generator in islice(self.create_text_generators(), 3, 5):
                with self.subTest(prompt=prompt, curr_generator=str(curr_generator)) as sub:
                    answer = curr_generator(
                        prompt_s=prompt,
                        max_new_tokens=curr_generator.group_size * 2,
                        return_full_text=True,
                    )["generated_text"]
                    self.assertIsInstance(answer, str, f"{answer} is not a string. "
                                                       f"curr_generator: {str(curr_generator)}")
                    self.assertGreater(len(answer), len(prompt), f"{answer} is not a not longer than "
                                                                 f"{prompt}. "
                                                                 f"curr_generator: {str(curr_generator)}")
                    self.assertTrue(answer.startswith(prompt[:-1]), f"{answer} doesn't start with "
                                                                    f"{prompt}. "
                                                                    f"curr_generator: {str(curr_generator)}")
                    print(f"ended sub test: {sub}")

    def test_post_process(self):
        """Tests the different returning options"""
        generator: TextGenerator = next(self.create_text_generators())
        prompt: str = "This is a test prompt"
        answer: Dict[str, Union[str, Tensor]] = generator(
            prompt_s=prompt,
            max_new_tokens=10,
            return_tensors=True,
            return_text=True,
            return_full_text=True
        )
        self.assertIsInstance(answer, dict,
                              f"{answer} is not a dict")
        self.assertIn("generated_text", answer.keys(),
                      f"{answer} doesn't contain 'generated_text'")
        self.assertIn("generated_token_ids", answer.keys(),
                      f"{answer} doesn't contain 'generated_token_ids'")
        self.assertIsInstance(answer["generated_text"], str,
                              f"{answer['generated_text']} is not a string")
        self.assertIsInstance(answer["generated_token_ids"], Tensor,
                              f"{answer['generated_token_ids']} is not a list")
        self.assertTrue(answer["generated_text"].startswith(prompt),
                        f"{answer['generated_text']} doesn't start with {prompt}")
        prompt_tokens: Tensor = generator.preprocess(prompt)
        self.assertIsInstance(prompt_tokens, Tensor,
                              f"{prompt_tokens} is not a tensor")
        self.assertIsInstance(answer["generated_token_ids"][:len(prompt_tokens)], Tensor,
                              f"{answer['generated_token_ids'][:len(prompt_tokens)]} is not a tensor")
        self.assertTrue(equal(answer["generated_token_ids"][:len(prompt_tokens)], prompt_tokens),
                        f"{answer['generated_token_ids'].tolist()} is not equal to {prompt_tokens}")

        answer: Dict[str, Union[str, Tensor]] = generator(
            prompt_s=prompt, max_new_tokens=10,
            return_tensors=True, return_text=True, return_full_text=False
        )
        self.assertFalse(answer["generated_text"].startswith(prompt),
                         f"{answer['generated_text']} doesn't start with {prompt}")
        prompt_tokens: Tensor = generator.preprocess(prompt)
        self.assertFalse(equal(answer["generated_token_ids"][:len(prompt_tokens)], prompt_tokens),
                         f"{answer['generated_token_ids'][:len(prompt_tokens)]} is not equal to {prompt_tokens}")

    def test_prefix(self):
        """Tests that the prefix option of the methods __call__ and preprocess works"""
        generator: TextGenerator = next(self.create_text_generators())
        prompt: str = "test prompt"
        prefix = "This is a"
        answer: Dict[str, Union[str, Tensor]] = generator(
            prompt_s=prompt, max_new_tokens=10,
            return_tensors=False, return_text=True, return_full_text=True, prefix=prefix
        )
        self.assertTrue(answer["generated_text"].startswith(prefix),
                        f"{answer['generated_text']} doesn't start with {prefix}")
        self.assertIn(prompt, answer["generated_text"], f"{answer['generated_text']} doesn't contain {prompt}")

    def test_num_return_sequences(self):
        """Tests that the num_return_sequences option of the methods __call__ and preprocess works"""
        tree_gen = TreeGenerator(model_name=self.MODEL_NAMES[0],
                                 group_size=2,
                                 top_k=3,
                                 top_p=None,
                                 end_of_sentence_stop=False)
        top_p_sampling_gen = SamplingGenerator(model_name=self.MODEL_NAMES[0],
                                               group_size=self.GROUP_SIZES[0],
                                               top_k=None,
                                               top_p=self.TOP_PS[0],
                                               temp=self.TEMPERATURES[0],
                                               end_of_sentence_stop=False)
        for generator in (tree_gen, top_p_sampling_gen):
            with self.subTest(generator=str(generator)) as sub:
                prompt: str = "I was to happy to see that "
                num_return_sequences = 2
                answer: List[Dict[str, Union[str, Tensor]]] = generator(
                    prompt_s=prompt, max_new_tokens=8, return_tensors=False, return_text=True,
                    return_full_text=True, num_return_sequences=num_return_sequences
                )
                self.assertEqual(len(answer), num_return_sequences, f"len(answer) is not {num_return_sequences}")
                for curr_answer in answer:
                    self.assertIn(prompt, curr_answer["generated_text"],
                                  f"{curr_answer['generated_text']} doesn't contain {prompt}")
                print(f"ended sub test: {sub}")

    def test_max_new_tokens_is_none(self):
        """test the __call__ function when max_new_tokens is None
        and end_of_sentence_stop is True"""
        tree_gen = TreeGenerator(model_name=self.MODEL_NAMES[0],
                                 group_size=2,
                                 top_k=1,
                                 top_p=None,
                                 end_of_sentence_stop=True)
        top_p_sampling_gen = SamplingGenerator(model_name=self.MODEL_NAMES[0],
                                               group_size=self.GROUP_SIZES[0],
                                               top_k=None,
                                               top_p=self.TOP_PS[0],
                                               temp=self.TEMPERATURES[0],
                                               end_of_sentence_stop=True)
        for generator in (top_p_sampling_gen, tree_gen):
            prompt: str = "This is a very short prompt"
            answer: Dict[str, Union[str, Tensor]] = generator(
                prompt_s=prompt, max_new_tokens=None,
                return_tensors=True, return_text=True, return_full_text=True
            )
            self.assertIsInstance(answer, dict,
                                  f"{answer} is not a dict")
            self.assertIn("generated_text", answer.keys(),
                          f"{answer} doesn't contain 'generated_text'")
            self.assertIn("generated_token_ids", answer.keys(),
                          f"{answer} doesn't contain 'generated_token_ids'")
            self.assertIsInstance(answer["generated_text"], str,
                                  f"{answer['generated_text']} is not a string")
            self.assertIsInstance(answer["generated_token_ids"], Tensor,
                                  f"{answer['generated_token_ids']} is not a list")
            self.assertTrue(answer["generated_text"].startswith(prompt),
                            f"{answer['generated_text']} doesn't start with {prompt}")
            prompt_tokens: Tensor = generator.preprocess(prompt)
            self.assertTrue(equal(answer["generated_token_ids"][:len(prompt_tokens)], prompt_tokens),
                            f"{answer['generated_token_ids'].tolist()} is not equal to {prompt_tokens}")


if __name__ == '__main__':
    main()

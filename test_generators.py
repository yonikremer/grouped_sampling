from unittest import TestCase, main
from typing import Generator

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
                top_p_sampling_gen.set_group_size(group_size)
                top_k_sampling_gen.set_group_size(group_size)
                curr_tree_gen.set_group_size(group_size)

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
            with self.subTest(curr_text_generator=str(curr_text_generator)):
                answer = curr_text_generator(self.PROMPT, curr_text_generator.group_size * 2)
                self.assertIsInstance(answer, str, f"{answer} is not a string. "
                                                   f"curr_text_generator: {str(curr_text_generator)}")
                self.assertGreater(len(answer), len(self.PROMPT), f"{answer} is not a not longer than {self.PROMPT}. "
                                                                  f"curr_text_generator: {str(curr_text_generator)}")
                self.assertTrue(answer.startswith(self.PROMPT), f"{answer} doesn't start with {self.PROMPT}. "
                                                                f"curr_text_generator: {str(curr_text_generator)}")

    repeating_prompt: str = "This is a very long text. " * 2

    def test_calling_generators_repeating_prompt(self):
        """Tests the generators with long inputs. (an edge case)"""
        special_generators = [TreeGenerator(model_name=self.MODEL_NAMES[0],
                                            group_size=self.GROUP_SIZES[0],
                                            top_k=None,
                                            top_p=1.0,
                                            temp=1000.0),
                              SamplingGenerator(model_name=self.MODEL_NAMES[0],
                                                group_size=self.GROUP_SIZES[0],
                                                top_k=None,
                                                top_p=1.0,
                                                temp=1000.0)
                              ]

        for curr_generator in special_generators:
            with self.subTest(curr_generator=str(curr_generator)):
                answer = curr_generator(self.repeating_prompt, curr_generator.group_size * 2)
                self.assertIsInstance(answer, str, f"{answer} is not a string. "
                                                   f"curr_generator: {str(curr_generator)}")
                self.assertGreater(len(answer), len(self.repeating_prompt), f"{answer} is not a not longer than "
                                                                            f"{self.repeating_prompt}. "
                                                                            f"curr_generator: {str(curr_generator)}")
                self.assertTrue(answer.startswith(self.repeating_prompt[:-1]), f"{answer} doesn't start with "
                                                                               f"{self.repeating_prompt}. "
                                                                               f"curr_generator: {str(curr_generator)}")


if __name__ == '__main__':
    main()

"""Contains the functions for the tokenizer."""

from typing import List
from functools import lru_cache

import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from flask import g  # A dictionary containing all the global variables.


def token_list_to_str(token_list: List[int], library: str) -> str:
    """decodes a list of tokens ids (integers) to a string"""
    if library in ("tensorflow", "tf"):
        token_ten = tf.convert_to_tensor(token_list, dtype=tf.int32)
        expanded = tf.expand_dims(token_ten, axis=0)
        tokenizer = g.tf_tokenizer
        detokenized = tokenizer.detokenize(expanded).to_tensor()
        bytes_ten = tf.squeeze(detokenized)
        bytes_list = bytes_ten.numpy().tolist()
        word_list: List[str] = [b.decode('utf-8') for b in bytes_list]
        str_ans =  " ".join(word_list).replace(" ##", "").replace("[PAD]", "")
        if "[END]" in str_ans:
            end_index = str_ans.index("[END]")
            str_ans = str_ans[:end_index]
        return str_ans

    elif library in ("torch", "pytorch", "pt"):
        tokenizer = g.pt_tokenizer
        word_list = tokenizer.decode(token_list, skip_special_tokens = True, clean_up_tokenization_spaces = False)
        return "".join(word_list)
    raise ValueError("library not recognized, must by a string and either 'tf', 'tensorflow', 'pt', 'torch', or 'pytorch'")


@lru_cache(maxsize = 10)
def create_tar(group_size):
    """Creates a tensor of shape [1, group_size]
    with random integers between 5 and the size of the vocabulary"""
    return tf.random.uniform([1, group_size], minval = 5, maxval = g.vocab_size, dtype = tf.int32)


def init_tf_tokenizer():
    """Creates and returns the tokenizer and vocabulary."""
    path = 'flaskr/static/look_up_table.txt'
    with open(path, 'r', encoding="utf-8") as file:
        vocab = file.read().split()
    tf_tokenizer = Tokenizer(lower_case = True).fit_on_texts(vocab)
    return tf_tokenizer, vocab


def tokenize_and_preprocess(text):
    """Converts string to tensor of tokens of shape [1, seq_length]."""
    ragged = g.tf_tokenizer.tokenize(text)[0, :]
    eager = ragged.to_tensor(default_value = 0, shape = [None, 1])
    # 0 is the value of the padding token
    squeezed = tf.squeeze(eager, axis = 1)
    typed = tf.cast(squeezed, tf.int32)
    start_token = tf.constant([2], dtype=tf.int32)
    edited = tf.concat([start_token, typed], 0)
    expended = tf.expand_dims(edited, axis=0)
    return expended

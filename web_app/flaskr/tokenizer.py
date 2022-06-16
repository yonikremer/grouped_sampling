"""Contains the functions for the tokenizer."""

import tensorflow as tf
import tensorflow_text as tf_text
from flask import g  # A dictionary containing all the global variables.
from typing import List


def init_tokenizer() -> tf_text.BertTokenizer:
    """Creates and returns the tokenizer."""
    path: str = 'flaskr/static/look_up_table.txt'
    bert_tokenizer_params: dict = dict(lower_case = True)
    with open(path, 'r') as f:
        vocab: List[str] = f.read().split()
    tensor_vocab: List[tf.Tensor] = [tf.convert_to_tensor(token_key, dtype = tf.string) for token_key in vocab]
    lookup_table = tf.lookup.StaticVocabularyTable(
        tf.lookup.KeyValueTensorInitializer(
            keys = tensor_vocab,
            key_dtype = tf.string,
            values = tf.range(tf.size(vocab, out_type = tf.int64), dtype = tf.int64),
            value_dtype = tf.int64),
        num_oov_buckets = 1
    )
    my_tokenizer: tf_text.BertTokenizer = tf_text.BertTokenizer(lookup_table, **bert_tokenizer_params)
    # tf_text.Tokenizer is the base class for tf_text.BertTokenizer
    return my_tokenizer


def tokenize(text: str) -> tf.Tensor:
    """Converts string to tensor"""
    ragged: tf.RaggedTensor = g.my_tokenizer.tokenize(text)[0, :]
    eager: tf.Tensor = ragged.to_tensor(default_value = 0, shape = [None, 1])  # 0 is the value of the padding token
    sqeezed: tf.Tensor = tf.squeeze(eager, axis = 1)
    typed: tf.Tensor = tf.cast(sqeezed, tf.int32)
    start_token = tf.constant([2], dtype=tf.int32)
    end_token = tf.constant([3], dtype=tf.int32)
    edited: tf.Tensor = tf.concat([start_token, typed, end_token], axis = 0)
    return edited

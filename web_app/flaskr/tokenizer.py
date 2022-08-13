"""Contains the functions for the tokenizer."""

import tensorflow as tf
from tensoeflow_text import BertTokenizer
from flask import g  # A dictionary containing all the global variables.


def init_tokenizer():
    """Creates and returns the tokenizer and vocabulary."""
    path = 'flaskr/static/look_up_table.txt'
    bert_tokenizer_params = dict(lower_case = True)
    with open(path, 'r', encoding="utf-8") as file:
        vocab = file.read().split()
    tensor_vocab = [tf.convert_to_tensor(token_key, dtype = tf.string) for token_key in vocab]
    lookup_table = tf.lookup.StaticVocabularyTable(
        tf.lookup.KeyValueTensorInitializer(
            keys = tensor_vocab,
            key_dtype = tf.string,
            values = tf.range(tf.size(vocab, out_type = tf.int64), dtype = tf.int64),
            value_dtype = tf.int64
            ),
        num_oov_buckets = 1
    )
    my_tokenizer = BertTokenizer(lookup_table, **bert_tokenizer_params)
    return my_tokenizer, vocab


def tokenize_and_preprocess(text):
    """Converts string to tensor of tokens of shape [1, seq_length]."""
    ragged = g.my_tokenizer.tokenize(text)[0, :]
    eager = ragged.to_tensor(default_value = 0, shape = [None, 1])
    # 0 is the value of the padding token
    squeezed = tf.squeeze(eager, axis = 1)
    typed = tf.cast(squeezed, tf.int32)
    start_token = tf.constant([2], dtype=tf.int32)
    edited = tf.concat([start_token, typed], 0)
    expended = tf.expand_dims(edited, axis=0)
    return expended

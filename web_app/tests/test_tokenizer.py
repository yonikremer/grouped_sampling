from flaskr import tokenizer

# from hypothesis import given, example
# from hypothesis.strategies import text
import tensorflow as tf
from tensorflow_text import BertTokenizer
from random import selection, randint
from typing import List


def test_init_tokenizer():
    my_tokenizer = tokenizer.init_tokenizer()
    assert isinstance(my_tokenizer, BertTokenizer)
    assert my_tokenizer.vocab_size <= 8192
    assert my_tokenizer.vocab_size >= 4096
    assert my_tokenizer.num_oov_buckets == 1
    assert my_tokenizer.lower_case


def test_tokenize_vocab():
    path: str = 'flaskr/static/look_up_table.txt'
    with open(path, 'r') as f:
        vocab: List[str] = f.read().split()
    new_vocab: List[str] = [t.replace('#', '') for t in vocab]
    my_tokenizer = tokenizer.init_tokenizer()
    for word in new_vocab:
        assert word in my_tokenizer.vocab
        assert tf.Tensor(vocab.index(word)) == tokenizer.tokenize_and_preprocces(word)


def test_tokenize_detokenize() -> None:
    path: str = 'flaskr/static/look_up_table.txt'
    with open(path, 'r') as f:
        vocab: List[str] = f.read().split()
    new_vocab: List[str] = [t.replace('#', '') for t in vocab]
    for _ in range(100):
        length: int = randint(0, 20)
        my_text: str = ' '.join([selection(new_vocab) for _ in range(length)])
        my_tokenizer = tokenizer.init_tokenizer()
        tokenized = my_tokenizer.tokenize(my_text)
        detokenized = my_tokenizer.detokenize(tokenized)
        assert detokenized == my_text
    unknown: str = "fbuefabieofcboie"
    assert "[UNK]" in my_tokenizer.tokenize(unknown)

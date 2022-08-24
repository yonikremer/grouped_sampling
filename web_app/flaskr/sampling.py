from typing import List, Dict, Tuple

from flask import g
from transformers import AutoModelForCausalLM
from torch.nn import Softmax
import tensorflow as tf

from tokenizer import tf_preprocess_and_tokenize, create_tar

def combinations(mat):
    """Returns all the lists such that list[j] is in mat[j]
    complexity: prod([len(mat[i]) for i in range(len(mat))])"""
    if len(mat) == 1:
        return [[mat[0][i]] for i in range(len(mat[0]))]
    res = []
    for i in mat[0]:
        for j in combinations(mat[1:]):
            res.append([i] + j)
    filtered_res = list(filter(doesnt_have_duplicates, res))
    return filtered_res


def doesnt_have_duplicates(my_list):
    """Return if there isn't a repetition in the list
    complexity: O(n) where n is the length of the list"""
    return len(my_list) == len(set(my_list))


def seq_prob(tokens, prob_mat, org_prompt_prob):
    """Given the probability matrix and a list of tokens
    returns the probability of the sequence
    prob_mat[a][b] is the probability of the token with id b the a-th token in the sequence"""
    probability = org_prompt_prob
    sequence_length = len(tokens)
    for i in range(sequence_length):
        curr_token = tokens[i]
        probability *= prob_mat[i][curr_token]
    return probability


def grouped_sampling(prob_mat: List[List[float]], top_p, top_k) -> List[List[int]]:
    """given a matrix of probabilities, returns a list of lists of tokens
    the matrixs is of size group_size x vocab_size
    where matrix[i, j] is the probability of token j the i-th token in the group
    samples the tokens such that for each place in the group,
    at most top_k tokens are sampled and at least one token is sampled
    and the added probability of all the tokens is less than or equal top_p
    returns a list of where every item is a tuple of a sequense and probability
    over all complexity of the function in O(group_size * vocab_size * log(vocab_size))"""
    
    # prob_tensor.shape is now (group_size, vocab_size)
    posible_tokens = []
    for token_prob in prob_mat: # group_size times
        vocab_size = len(token_prob)  # O(1)
        indexed_prob = list(zip(token_prob, range(vocab_size)))  # O(vocab_size)
        sorted_indexed_prob = sorted(indexed_prob, key=lambda x: x[0], reverse=True)[:top_k] # O(vocab_size*log(vocab_size))
        total_prob = 0
        curr_indices = []
        for prob, token in sorted_indexed_prob:  # O(top_k)
            if total_prob + prob > top_p:
                break
            total_prob += prob
            curr_indices.append(token)
        posible_tokens.append(curr_indices)  # O(1)
    new_sequences: List[List[int]] = combinations(posible_tokens)  # theta(prod(len(indices[i]) for i in range(group_size)))
    # len(indices[i]) < min(top_k, vocab_size)
    # therefore the complexity is O(min(top_k, vocab_size) * group_size)
    return new_sequences


def get_prob_mat(model_name, prompt, group_size):
    """Returns the probability matrix as a list of lists of floats"""
    model = g.loaded_models[model_name]
    tokenizer = g.loaded_tokenizers[model_name]
    if isinstance(model, AutoModelForCausalLM): 
        inputs = tokenizer(prompt, return_tensors="pt")
        if "bloom" in model_name:
            logits_pt_tensor = model(**inputs, labels=inputs["input_ids"]).logits.squeeze(0)
        else:
            logits_pt_tensor = model(**inputs).logits.squeeze(0)
        prob_tensor = Softmax(dim=1)(logits_pt_tensor)
        try:
            prob_tensor = prob_tensor[:group_size, :]
        except IndexError:
            pass
        
        prob_mat = [prob_tensor[i, :].tolist() for i in range(group_size)]
    else:
        tokenized_prompt = tf_preprocess_and_tokenize(prompt)
        tar = create_tar(group_size)
        prob_ten = model([tokenized_prompt, tar], training=False)
        prob_squeezed = tf.squeeze(prob_ten)
        prob_ndarray = tf.make_ndarray(prob_squeezed)
        prob_mat = [prob_ndarray[i, :].tolist() for i in range(group_size)]
    return prob_mat


def flatten(l: list) -> List:
    """Gets a list where some of the elements might be lists
    and adds every item in the inner list to the outer list.
    example: [1, [2, 3], 4, [[5]]] -> [1, 2, 3, 4, 5]
    Complexity: O(len(the flatten list) + the number of diffrent lists))"""
    new_list = []
    for item in l:
        if isinstance(item, list):
            new_list.extend(flatten(item))
        else:
            new_list.append(item)
    return new_list


def complete(model_name, org_prompt, top_p, top_k, num_groups, group_size, org_prompt_prob = 1.0) -> Dict[Tuple[int], float]:
    """preprocess the prompt and completes the text
    model name: str
    org_prompt: str
    top_p: float (0.0 - 1.0)
    top_k: int > 0
    num_groups: int > 0
    group_size: int > 0
    org_prompt_prob: float <= 1
    """
    tokenizer = g.loaded_tokenizers[model_name]
    if isinstance(org_prompt, str):
        str_prompt = org_prompt
        if "bloom" in model_name:
            tokenized_prompt_ten = tokenizer(str_prompt, return_tensors="pt")["input_ids"]
        else:
            tokenized_prompt_ten = tokenizer(str_prompt, return_tensors="pt")
        tokenized_prompt_list = tokenized_prompt_ten.tolist()
    elif isinstance(org_prompt, list):
        tokenized_prompt_list = flatten(org_prompt)
    else:
        raise ValueError("org_prompt must be a string or list of integers")
    
    prob_mat = get_prob_mat(model_name, str_prompt, group_size)
    tokenized_ans_list = grouped_sampling(prob_mat, top_p, top_k)
    prob_list: List[Tuple[List[int], float]] = [seq_prob(seq, prob_mat, org_prompt_prob) for seq in tokenized_ans_list]
    new_prompts: List[List[int]] = [tokenized_prompt_list + ans for ans in tokenized_ans_list]
    complition_prob_dict = remove_duplicates(new_prompts, prob_list)
    if num_groups == 1:
        return complition_prob_dict
    new_complitions: Dict[Tuple[int], float] = dict()
    for curr_new_prompt, curr_new_prompt_prob in complition_prob_dict.items():
        curr_completions: Dict[Tuple[int], float] = complete(model_name, curr_new_prompt, top_p, top_k, num_groups - 1, group_size, curr_new_prompt_prob)
        tokens: Tuple[int]
        prob: float
        for tokens, prob in curr_completions.items():
            new_complitions[tokens] = prob
    return new_complitions


def remove_duplicates(completions: List[List[int]], probs: List[float]) -> Dict[Tuple[int], float]:
    """Given a list of tokenized answers and the probability of each complition,
    removes every repeated completion and every complition that have repeated tokens"""
    filtered_completions: Dict[Tuple[int], float] = dict()
    for curr_comp, curr_prob in zip(completions, probs):
        if len(curr_comp) == len(set(curr_comp)):
            curr_comp_tuple = tuple(curr_comp)
            filtered_completions[curr_comp_tuple] = curr_prob
    return filtered_completions

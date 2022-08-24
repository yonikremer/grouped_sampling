from typing import List


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

import ipdb
from typing import List, Set

def compute_shortest_tokenizations(
    base_representation_s: List[str], 
    vocabulary: Set[str], 
    disregard_word_initial_marker: bool, 
    word_initial_marker: str
     ) -> List[List[str]]:
    """Compute the shortest tokenization of the input string using the provided vocabulary.

    Args:
        base_representation_s (List[str]): The input string to tokenize, represented as a list of atomic tokens
            (e.g., characters for Unigram, bytes for Byte-level BPE, etc.)
        vocabulary (Set[str]): The set of all tokens in the vocabulary.
        disregard_word_initial_marker (bool): Whether to disregard the word initial marker. This means 
            that if we have two tokens that are only different in that one has the word initial marker and the other
            does not, we will treat them as the same token. Set this to true for encoding. But for decoding, 
            you'd want to set this to false in order to be able to decode to a unique, orthographically readable string.
        word_initial_marker (str): The word initial marker. (e.g., '##')
    """
    if disregard_word_initial_marker:
        vocabulary = {token.lstrip(word_initial_marker) for token in vocabulary}

    ipdb.set_trace()
    n = len(base_representation_s)
    len_dp = [float('inf')] * (n + 1)
    len_dp[0] = 0  # Base case: empty string
    segment_index_dp = [[i] for i in range(n)]
    
    def _get_concatenation(slice: List[str]):
        return "".join(slice)

    for i in range(1, n + 1):
        min_split_indices = []
        prev_min_length = len_dp[i]
        for j in range(i):
            if _get_concatenation(base_representation_s[j:i]) in vocabulary:
                if len_dp[i] > len_dp[j] + 1:
                    len_dp[i] = len_dp[j] + 1
                    if len_dp[i] < prev_min_length:
                        min_split_indices = [j]
                if len_dp[i] == len_dp[j] + 1:
                    if j not in min_split_indices: # we should be able to do without this, idk
                        min_split_indices.append(j)
        segment_index_dp[i-1] = min_split_indices
    
    # do backtracking to get all the tokenizations with the minimum number of tokens, using segment_index_dp
    backtrace_indices = segment_index_dp[-1]
    curr_index = n - 1
    tokenizations = []
    curr_tokenization = []
    while backtrace_indices:
        backtrace_index = backtrace_indices.pop()
        curr_tokenization.insert(0, ''.join(base_representation_s[backtrace_index:curr_index + 1]))
        curr_index = backtrace_index - 1
        if curr_index == -1:
            tokenizations.append(curr_tokenization)
            curr_tokenization = []
            curr_index = n - 1
        else:
            backtrace_indices.extend(segment_index_dp[curr_index])
    return tokenizations, len_dp[-1]

def obtain_longest_token(tokenizations: List[List[str]]) -> List[str]:
    """Obtain the longest tokenization from a list of tokenizations.

    Args:
        tokenizations (List[List[str]]): List of tokenizations to choose from.

    Returns:
        List[str]: The longest tokenization.
    """
    tokenization_lengths = [len(tokenization) for tokenization in tokenizations]
    # return the tokenization with the longest length
    return tokenizations[tokenization_lengths.index(max(tokenization_lengths))]
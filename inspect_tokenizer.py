from typing import List, Dict, Set
import ipdb

# def step_get_vocab_and_merges(tokenizer: ByteLevelBPETokenizer, 
#                               save_fname: str, 
#                               **kwargs) -> Tuple[Dict[str, int], List[str]]:
#     vocab = tokenizer.get_vocab()
#     tokenizer.save(save_fname) 
#     # open f"{save_fname}.json" as a json object
#     with open(f"{save_fname}.json") as f:
#         vocab = json.load(f)
#     merges = vocab['model']['merges'] # A list of strings. Each string is of the form "{token} {token}", showing the two tokens that were merged to create a new one 
#     token_to_i = vocab['model']['vocab'] # a dictionary mapping tokens to their indices (integers) in the model. 
#     ipdb.set_trace()
#     return token_to_i, merges

def obtain_token_compositions(token_str: str, vocab: Dict[str, int], merges: List[str]) -> List[List[str]]:
    """Obtain all the ways to construct the token_str from the vocabulary.

    Args:
        token_str (str): Some token string (e.g., Êģ)
        vocab (Dict[str, int]): All the tokens in the vocabulary
        merges (List[str]): All the merges in the vocabulary
    """
    if len(token_str) == 1: # TODO: maybe using len here is not right
        return [[token_str]]
    decompositions = []    
    for i in range(len(token_str)):
        left = token_str[:i]
        right = token_str[i:]
        if f"{left} {right}" in merges:
            assert (left in vocab) and (right in vocab)
            decompositions.append([left, right])
            if len(left) > 1:
                decompositions.extend(
                    [left_decomposition + [right] for left_decomposition in obtain_token_compositions(left, vocab, merges)]
                )
            if len(right) > 1:
                decompositions.extend(
                    [[left] + right_decomposition for right_decomposition in obtain_token_compositions(right, vocab, merges)]
                )
    return decompositions

def compute_length_of_most_efficient_tokenization(sequence: List[str], vocabulary: Set[str]):
    """Compute the least number of tokens required to represent {sequence} using tokens from
    the {vocabulary}. For example, if {sequence} is *in* the vocabulary, then the most efficient
    tokenization has a length of 1. 

    Args:
        sequence (List[str]): the sequence represented as a list of characters. E.g., for a sequence like "the weather", 
            it will be provided as an input to this function as ['t', 'h', 'e', ' ', 'w', 'e', 'a', 't', 'h', 'e', 'r'].  
            You can assume each of the individual characters are in the vocabulary (e.g., s[i], for i in range(len(sequence))).
            However, s[i:j] for j>i is not guaranteed to be in the vocabulary. 
        vocabulary: All of the tokens in the model's vocabulary. For HuggingFace models, obtaining this vocabulary is easy.
            FOr example, for Bloom-3B, have a look at tokenizer.json: https://huggingface.co/bigscience/bloom-3b/tree/main. 
    """
    # TODO: implement the algorithm described in: https://www.notion.so/farhansamir/Understanding-byte-level-BPE-tokenization-dcba9b82078b4720be06efc6cbe14a90?pvs=4.
    # TODO: after it is implemented, test it out by comparing the length provided this function and see whether it is shorter than the number of tokens provided by
        # using `tokenizer.encode("the weather")`. If this check fails, there are two possibilities. 1. The implementation is wrong. 2. Our logic for the algorithm is wrong. (2) is unlikely 
    pass

def min_tokens_for_string(base_representation_s: List[str], vocabulary: Set[str]):
    n = len(base_representation_s)
    dp = [float('inf')] * (n + 1)
    dp[0] = 0  # Base case: empty string
    
    def _get_concatenation(slice: List[str]):
        return "".join(slice)

    for i in range(1, n + 1):
        for j in range(i):
            if _get_concatenation(base_representation_s[j:i]) in vocabulary:
                dp[i] = min(dp[i], dp[j] + 1)
    return dp[n]


def min_tokens_for_string(s: str, vocabulary: Set[str]):
    n = len(s)
    dp = [float('inf')] * (n + 1)
    dp[0] = 0  # Base case: empty string
    
    for i in range(1, n + 1):
        for j in range(i):
            if s[j:i] in vocabulary:
                dp[i] = min(dp[i], dp[j] + 1)
    return dp[n]

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
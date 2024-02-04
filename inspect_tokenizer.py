import ipdb

def step_get_vocab_and_merges(tokenizer: ByteLevelBPETokenizer, 
                              save_fname: str, 
                              **kwargs) -> Tuple[Dict[str, int], List[str]]:
    vocab = tokenizer.get_vocab()
    tokenizer.save(save_fname) 
    # open f"{save_fname}.json" as a json object
    with open(f"{save_fname}.json") as f:
        vocab = json.load(f)
    merges = vocab['model']['merges'] # A list of strings. Each string is of the form "{token} {token}", showing the two tokens that were merged to create a new one 
    token_to_i = vocab['model']['vocab'] # a dictionary mapping tokens to their indices (integers) in the model. 
    ipdb.set_trace()
    return token_to_i, merges

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
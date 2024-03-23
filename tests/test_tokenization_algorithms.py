import ipdb
from transformers import AutoTokenizer
from inspect_tokenizer import min_tokens_for_string, compute_shortest_tokenizations
from packages.tokenizer_utils import dp_tokenize_llama

def test_quadratic_algorithm():
    vocabulary = {"a", "ab", "abc", "d", "cd", "bcd", "b", "c"}
    input_string = "abcd"
    assert min_tokens_for_string(input_string, vocabulary) == 2
    input_string_two = "adcbdab"
    assert min_tokens_for_string(input_string_two, vocabulary) == 6

    input_string = "abdcd"
    assert min_tokens_for_string(input_string, vocabulary) == 3

    vocabulary = {"un", "desirable", "und", "es", "ira", "ble", "u", "n"}
    input_string = "undesirable"
    assert min_tokens_for_string(input_string, vocabulary) == 2

    vocabulary = {"un", "desirable", "und", "es", "ira", "ble", "ish"}
    input_string = "undesirableish"
    assert min_tokens_for_string(input_string, vocabulary) == 3

def test_compute_shortest_tokenization():
    V = ["un", "desirable"] + ["und", "esirable"] + list("undesirable")
    input_string = "undesirable"
    shortest_tokenizations, shortest_length = compute_shortest_tokenizations(input_string, V, False, "") 
    print(shortest_length)
    print(f"\n{shortest_tokenizations}")
    assert ["un", "desirable"] in shortest_tokenizations
    assert ["und", "esirable"] in shortest_tokenizations

def test_llama_tokenizer():
    # TODO: watch out for multi-sentence encoding 
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir="hf_cache")
    input_string = "the weather"
    encoding = tokenizer.encode(input_string)
    t2i_dict = tokenizer.get_vocab()
    i2d_dict = {v: k for k, v in t2i_dict.items()}
    space_token = '▁'
    base_representation_s = list(input_string)
    # replace ' ' with a space
    # base_representation_s = [space_token if char == ' ' else char for char in base_representation_s]
    # replace with for loop
    for i, char in enumerate(base_representation_s):
        if i == 0:
            base_representation_s[i] = space_token + char
        if char == ' ':
            base_representation_s[i] = space_token

    shortest_tokenizations, length = compute_shortest_tokenizations(base_representation_s, set(t2i_dict.keys()), False, None)
    min_tokenization = shortest_tokenizations[0]
    for i in range(len(min_tokenization)):
        if i == 0:
            # strip the space token
            min_tokenization[i] = min_tokenization[i].lstrip(space_token)
        else:
            min_tokenization[i] = min_tokenization[i].replace(space_token, ' ')
    min_string = ''.join(min_tokenization)
    assert min_string == "the weather"
    assert length == 2
# def select_by_longest_token(tokenizations: List[List[str]]):
#     tokenization_lengths = [len(tokenization) for tokenization in tokenizations]
#     # return the tokenization with the longest length
#     return tokenizations[tokenization_lengths.index(max(tokenization_lengths))]
def test_llama_tokenize_reversal():
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir="hf_cache")
    i2t = {v: k for k, v in tokenizer.get_vocab().items()}
    space_token = '▁'
    t2i = tokenizer.get_vocab()
    dp_encode_llama, invert_dp_tokenize = dp_tokenize_llama(tokenizer)
    input_string = "the weather"
    encoding = dp_encode_llama(input_string)
    decoded_string = invert_dp_tokenize(encoding)
    assert decoded_string == "the weather"

    long_string = 'General thermodynamic relations for the work of polydisperse micelle formation in the model of ideal solution of molecular aggregates in nonionic surfactant solution and the model of "dressed micelles" in ionic solution have been considered.'
    default_encoding = tokenizer.encode(long_string)[1:]
    encoding = dp_encode_llama(long_string)
    decoded_string = invert_dp_tokenize(encoding)
    assert decoded_string == long_string
    assert len(encoding) <= len(default_encoding)
    print(f"Default encoding length: {len(default_encoding)}")
    print(f"DP encoding length: {len(encoding)}")
    dp_encoding = [i2t[t] for t in encoding]
    default_encoding = [i2t[t] for t in default_encoding]

    for token in dp_encoding:
        if token not in default_encoding:
            print(f"Token: {token} not in default encoding")
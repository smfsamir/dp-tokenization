from bidict import bidict
import ipdb
from typing import List, Set
from .dp_tokenize import compute_shortest_tokenizations, obtain_longest_token

def merge_tokens(tokens, sep='Ġ'):
    merged_tokens = []
    i = 0
    while i < len(tokens):
        # Merge tokens that are part of the same word
        if i + 1 < len(tokens) and not tokens[i + 1].startswith(sep):
            merged_token = tokens[i]
            while i + 1 < len(tokens) and not tokens[i + 1].startswith(sep):
                merged_token += tokens[i + 1]
                i += 1
            merged_tokens.append(merged_token)
        else:
            merged_tokens.append(tokens[i])
        i += 1

    return merged_tokens

def pretokenize_with_llama(tokenizer, vocab_bidict):
    def pretokenize(input_str):
        tokens = tokenizer.encode(input_str)
        tokens = [vocab_bidict.inverse[token] for token in tokens]
        space_token = '▁'
        pre_tokens = merge_tokens(tokens, sep=space_token)
        return pre_tokens
    return pretokenize

def pretokenize_raw(manual_mapping) -> List[List[str]]:
    def pretokenize(input_str):
        base_representation_s = list(input_str)
        space_token = '▁'
        pretokenized = []
        prev_segment_index = 0
        for i, char in enumerate(base_representation_s):
            if i == 0:
                base_representation_s[i] = space_token + char
            elif char == ' ':
                base_representation_s[i] = space_token
                pretokenized.append(base_representation_s[prev_segment_index:i ])
                prev_segment_index = i 
            elif char in manual_mapping.inverse:
                base_representation_s[i] = manual_mapping.inverse[char]
        pretokenized.append(base_representation_s[prev_segment_index:])
        return pretokenized
    return pretokenize

def dp_tokenize_llama(llama_tokenizer, pretokenize_option = 'llama'):
    t2i_dict = llama_tokenizer.get_vocab()
    i2t_dict = {v: k for k, v in t2i_dict.items()}
    vocab_bidict = bidict(t2i_dict)
    space_token = '▁'
    vocab = set(llama_tokenizer.get_vocab())

    manual_mapping = bidict({
        '<0x0A>': '\n'
    })
    if pretokenize_option == 'raw':
        pretokenize_func = pretokenize_raw(manual_mapping)
    elif pretokenize_option == 'llama':
        pretokenize_func = pretokenize_with_llama(llama_tokenizer, vocab_bidict)
    def dp_tokenize(input_str) -> List[int]:
        # TODO: we need to pre-tokenize the input string
        pretokenized = pretokenize_func(input_str)
        selected_tokenizations = []
        for pretokenized_seq in pretokenized:
            shortest_tokenizations, length = compute_shortest_tokenizations(pretokenized_seq, vocab, False, None, 1)
            if not shortest_tokenizations:
                ipdb.set_trace()
            selected_tokenization = obtain_longest_token(shortest_tokenizations)
            selected_tokenizations.append(selected_tokenization)
        encoded_tokenization = []
        for tokenization in selected_tokenizations:
            for token in tokenization:
                encoded_tokenization.append(t2i_dict[token])
        print(encoded_tokenization)
        return encoded_tokenization
    
    def decode_dp_tokenization(encoding: List[int]):
        # first, decode the encoding using the i2t_dict
        return llama_tokenizer.decode(encoding)[4:]
        # decoded_tokens = [i2t_dict[token] for token in encoding][1:]
        # for i in range(len(decoded_tokens)):
        #     if i == 0:
        #         # strip the space token
        #         decoded_tokens[i] = decoded_tokens[i].lstrip(space_token)
        #     elif decoded_tokens[i] in manual_mapping:
        #         decoded_tokens[i] = manual_mapping[decoded_tokens[i]]
        #     else:
        #         decoded_tokens[i] = decoded_tokens[i].replace(space_token, ' ')
        # decoded_string = ''.join(decoded_tokens)
        # return decoded_string
    return dp_tokenize, decode_dp_tokenization
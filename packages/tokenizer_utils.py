import json
from bidict import bidict
import ipdb
from typing import List, Set, Dict
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

def dp_tokenize_bloom(bloom_tokenizer, HF_CACHE_DIR):
    def get_merge_list(tokenizer_json_fname: str):
        with open(tokenizer_json_fname, 'r') as f:
            tokenizer_json = json.load(f)
            merges = tokenizer_json['model']['merges']
            return merges
    
    def get_token_to_index_map(tokenizer_json_fname) -> Dict[str, int]:
        with open(tokenizer_json_fname, 'r') as f:
            tokenizer_json = json.load(f)
            vocab = tokenizer_json['model']['vocab']
            return {token: index for index, token in enumerate(vocab)}

    tokenizer_json_fname = f"{HF_CACHE_DIR}/models--bigscience--bloom-3b/snapshots/52bc5b43010b4844513826b8be3f78c7344c37d7/tokenizer.json"
    merge_list = get_merge_list(tokenizer_json_fname)
    vocab_to_index = bidict(get_token_to_index_map(tokenizer_json_fname))


    token_to_source_merge = {}
    for merge in merge_list:
        a, b = merge.split()
        token_to_source_merge[a + b] = (a, b)
    
    def unwind_to_base_tokenization(input_str: str) -> List[int]: 
        """
        Params:
            input_str (str): String to get the base tokenization for.
            vocab_token_to_index (Dict[str, int]): Mapping from token to index in the vocabulary.
            vocab_merges (List[str]): List of merges in the vocabulary. Of the form "a b" where a and b are tokens.
            tokenizer: Huggingface tokenizer object.
        """
        encoding = bloom_tokenizer.encode(input_str) # List[int]
        # create an object token_to_source_merge. This will be a dictionary that maps a token (str) to the merge that created it (Tuple[str,str]).
        base_tokens = []

        def decompose(token: str) -> List[int]:
            if token in token_to_source_merge:
                a, b = token_to_source_merge[token]
                return decompose(a) + decompose(b) 
            else:
                # base token
                return [vocab_to_index[token]]

        base_tokens = []
        for token_ind in encoding:
            token = bloom_tokenizer.convert_ids_to_tokens([token_ind])[0]
            base_tokens.extend(decompose(token))
        return base_tokens

    def compute_shortest_tokenization_bloom(input_str):
        # token_inds = unwind_to_base_tokenization(input_str) # get the longest tokenization (i.e., encoding using the alphabet only)
        token_inds = [vocab_to_index[c] for c in input_str]
        tokens = bloom_tokenizer.convert_ids_to_tokens(token_inds) # 
        return compute_shortest_tokenizations(tokens, vocab_to_index, False, "Ġ") # this is the DP algorithm
    
    # def min_tokens_for_string(tokens: List[str]):
    #     _, shortest_length = compute_shortest_tokenization_bloom(tokens)
    #     return shortest_length

    def pretokenize(input_str):
        pretokenized = bloom_tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str(input_str)
        return [pretokenized_block[0] for pretokenized_block in pretokenized]
    
    def dp_tokenize(input_str) -> List[int]:
        pretokenized = pretokenize(input_str) # List[str]
        selected_tokenizations = []

        for pretokenized_seq in pretokenized: # iterate over the words in the sentence {input_str}
            shortest_tokenizations, length = compute_shortest_tokenization_bloom(pretokenized_seq)  # compute the shortest tokenization for the word
            selected_tokenization = obtain_longest_token(shortest_tokenizations)
            selected_tokenizations.append(selected_tokenization)
        
        encoded_tokenization = []
        for tokenization in selected_tokenizations:
            for token in tokenization:
                encoded_tokenization.append(vocab_to_index[token])
        return encoded_tokenization

    def decode_dp_tokenization(encoding: List[int]):
        # decoded_tokens = [vocab_to_index.inverse[token] for token in encoding]
        # decoded_string = ''.join(decoded_tokens)
        return bloom_tokenizer.decode(encoding)

    return dp_tokenize, decode_dp_tokenization
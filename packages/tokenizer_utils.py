import ipdb
from typing import List, Set
from .dp_tokenize import compute_shortest_tokenizations, obtain_longest_token

def dp_tokenize_llama(llama_tokenizer):
    t2i_dict = llama_tokenizer.get_vocab()
    i2t_dict = {v: k for k, v in t2i_dict.items()}
    space_token = '▁'
    vocab = set(llama_tokenizer.get_vocab())
    def dp_tokenize(input_str) -> List[int]:
        base_representation_s = list(input_str)
        for i, char in enumerate(base_representation_s):
            if i == 0:
                base_representation_s[i] = space_token + char
            if char == ' ':
                base_representation_s[i] = space_token
        shortest_tokenizations, length = compute_shortest_tokenizations(base_representation_s, vocab, False, None)
        selected_tokenization = obtain_longest_token(shortest_tokenizations)
        encoded_tokenization = [llama_tokenizer.bos_token_id]
        for token in selected_tokenization:
            encoded_tokenization.append(t2i_dict[token])
        return encoded_tokenization
    
    def decode_dp_tokenization(encoding: List[int]):
        # first, decode the encoding using the i2t_dict
        decoded_tokens = [i2t_dict[token] for token in encoding][1:]
        for i in range(len(decoded_tokens)):
            if i == 0:
                # strip the space token
                decoded_tokens[i] = decoded_tokens[i].lstrip(space_token)
            else:
                decoded_tokens[i] = decoded_tokens[i].replace(space_token, ' ')
        decoded_string = ''.join(decoded_tokens)
        return decoded_string
    return dp_tokenize, decode_dp_tokenization
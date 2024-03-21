import numpy as np
import ipdb
import polars as pl
from typing import Dict, Set
import pandas as pd
from functools import partial
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, WhisperTokenizer

from inspect_tokenizer import compute_shortest_tokenizations

# create a dir called hf_cache in the current directory
os.makedirs("hf_cache", exist_ok=True)
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-3b", cache_dir="hf_cache", use_fast=False)
whisper_tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny", task="transcribe", cache_dir='hf_cache')

bloom_vocab = tokenizer.get_vocab()

t2i = tokenizer.convert_tokens_to_ids
i2t = tokenizer.convert_ids_to_tokens

frame = pl.from_pandas(pd.read_csv("madar_lexicon.tsv", sep='\t'))


import json
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

    
tokenizer_json_fname = "hf_cache/models--bigscience--bloom-3b/snapshots/52bc5b43010b4844513826b8be3f78c7344c37d7/tokenizer.json"
merge_list = get_merge_list(tokenizer_json_fname)
vocab_to_index = get_token_to_index_map(tokenizer_json_fname)

token_to_source_merge = {}
for merge in merge_list:
    a, b = merge.split()
    token_to_source_merge[a + b] = (a, b)


from typing import Dict, List

def unwind_to_base_tokenization(input_str: str) -> List[int]: 
    """
    Params:
        input_str (str): String to get the base tokenization for.
        vocab_token_to_index (Dict[str, int]): Mapping from token to index in the vocabulary.
        vocab_merges (List[str]): List of merges in the vocabulary. Of the form "a b" where a and b are tokens.
        tokenizer: Huggingface tokenizer object.
    """
    encoding = tokenizer.encode(input_str) # List[int]
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
        token = tokenizer.convert_ids_to_tokens([token_ind])[0]
        base_tokens.extend(decompose(token))
    return base_tokens

def min_tokens_for_string(tokens: List[str]):
    _, shortest_length = compute_shortest_tokenizations(tokens, bloom_vocab, False, "Ġ")
    return shortest_length

def compute_min_tokenization(input_str):
    token_inds = unwind_to_base_tokenization(input_str)
    tokens = tokenizer.convert_ids_to_tokens(token_inds)
    return min_tokens_for_string(tokens)

def compute_default_tokenization_length(input_str):
    return len(tokenizer.encode(input_str))


# frame_sample = frame.filter(pl.col('CODA').str.len_chars())
frame_sample = frame 
strings = frame_sample['CODA']

min_lengths = []
for s in tqdm(strings):
    min_length = compute_min_tokenization(s)
    min_lengths.append(min_length)

default_lengths = []
for s in tqdm(strings):
    default_length = compute_default_tokenization_length(s)
    default_lengths.append(default_length)
frame = frame.with_columns([
    pl.Series(name='min_length', values=min_lengths),
    pl.Series(name='default_length', values=default_lengths)
    # pl.lit(min_lengths).alias('min_length'),
    # pl.lit(default_lengths).alias('default_length')
])
frame = frame.with_columns([
    pl.col('MSA').map_elements(compute_default_tokenization_length).alias('default_msa'),
    pl.col('MSA').map_elements(compute_min_tokenization).alias('min_msa')
])
# print(np.array(min_lengths).mean())
# print(np.array(default_lengths).mean())
ipdb.set_trace()
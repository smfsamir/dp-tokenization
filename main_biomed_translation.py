import json
from dotenv import load_dotenv
import os
from typing import List, Dict
import polars as pl
import subprocess
from flowmason import SingletonStep, MapReduceStep
from collections import OrderedDict
from transformers import AutoTokenizer, WhisperTokenizer


load_dotenv()
WMT_SAVE_DIR = os.getenv("WMT_DATASET_SAVE_DIR") # create .env file in root and set this to whereever to save
HF_CACHE_DIR = os.getenv("HF_CACHE_DIR") # create .env file in root and set this to whereever to save

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

def step_download_datasets(language_pair, 
                           **kwargs):
    """
    """
    # https://github.com/biomedical-translation-corpora/corpora
    # git clone the repo to the WM_SAVE_DIR using subprocess
    assert f"{WMT_SAVE_DIR}/biomedical-translation-corpora" in os.listdir(WMT_SAVE_DIR), f"The biomed translation corpus project needs to be cloned to {WMT_SAVE_DIR}"
    return True

def unwind_to_base_tokenization(tokenizer, input_str: str) -> List[int]: 
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

def compute_shortest_tokenization_bloom(input_str):
    token_inds = unwind_to_base_tokenization(input_str)
    tokens = tokenizer.convert_ids_to_tokens(token_inds)
    return min_tokens_for_string(tokens)

def step_compare_dp_default_tokenization(dataset_path, 
                                        language_pair: str,
                                         model_tokenizer) -> pl.DataFrame:
    """

    Returns a dataframe containing the same number of
    rows as the input dataset with the following columns:
    - src text
    - target text
    - dp number of tokens
    - default number of tokens
    - default tokenization strings (List[str]).
    - DP tokenization strings (List[str]).
    """
    tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-3b", cache_dir=HF_CACHE_DIR, use_fast=False)
    pass


if __name__ == '__main__':
    steps = OrderedDict()
    steps['download_biomed_dataset'] = SingletonStep(step_download_datasets, {
        'language_pair': 'en-de',
        'version': '001'
    })

    steps['compare_dp_default_tokenization'] = SingletonStep(step_compare_dp_default_tokenization, {
        'dataset_path': f"{WMT_SAVE_DIR}/biomedical-translation-corpora/data/en-de/train.en",
        'language_pair': 'en-de',
        'model_tokenizer': 'bloomz/mbart-large-cc25'
    })

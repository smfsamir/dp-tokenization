import json
from dotenv import load_dotenv
import ipdb
import os
from typing import List, Dict
import polars as pl
import subprocess
from flowmason import SingletonStep, MapReduceStep
from collections import OrderedDict
from transformers import AutoTokenizer, WhisperTokenizer
import loguru

from packages.tokenizer_utils import dp_tokenize_llama, dp_tokenize_bloom

logger = loguru.logger

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
    assert f"{WMT_SAVE_DIR}/29212094_en.txt" in os.listdir(WMT_SAVE_DIR), f"The biomed translation corpus project needs to be cloned to {WMT_SAVE_DIR}"
    return True

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
    tokenizer = AutoTokenizer.from_pretrained(model_tokenizer, cache_dir=HF_CACHE_DIR)
    dp_encode_bloom, invert_dp_tokenize = dp_tokenize_bloom(tokenizer, HF_CACHE_DIR)

    def assert_action(msg):
        logger.error(msg)
        ipdb.set_trace()

    for sample_txt_fname in os.listdir(dataset_path):
        with open(f"{dataset_path}/{sample_txt_fname}") as f:
            txt = f.read().strip()
            default_tokenizer_length = len(tokenizer.encode(txt))
            dp_encoded_text = dp_encode_bloom(txt)
            dp_tokenizer_length = len(dp_encoded_text)
            assert default_tokenizer_length >= dp_tokenizer_length, assert_action(f"DP tokenization is longer than default tokenization for {txt}")
            assert invert_dp_tokenize(dp_encoded_text) == txt, assert_action(f"DP tokenization is not reversible for {txt}")
    ipdb.set_trace()


if __name__ == '__main__':
    steps = OrderedDict()
    steps['download_biomed_dataset'] = SingletonStep(step_download_datasets, {
        'language_pair': 'en-de',
        'version': '001'
    })

    steps['compare_dp_default_tokenization'] = SingletonStep(step_compare_dp_default_tokenization, {
        'dataset_path': f"{WMT_SAVE_DIR}",
        'language_pair': 'en-de',
        'model_tokenizer': 'bigscience/bloom-3b'
    })
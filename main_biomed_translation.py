import json
from dotenv import load_dotenv
import ipdb
import os
from typing import List, Dict
import pandas as pd
import polars as pl
import subprocess
from flowmason import SingletonStep, MapReduceStep, conduct
from collections import OrderedDict
from datasets import Dataset
from transformers import AutoTokenizer, WhisperTokenizer
import loguru
from tqdm import tqdm

from packages.tokenizer_utils import dp_tokenize_llama, dp_tokenize_bloom
from packages.constants import SCRATCH_DIR

logger = loguru.logger

load_dotenv()
WMT_SAVE_DIR = os.getenv("WMT_DATASET_SAVE_DIR") # create .env file in root and set this to whereever to save
HF_CACHE_DIR = os.getenv("HF_CACHE_DIR") # create .env file in root and set this to whereever to save


def step_download_datasets(language_pair, 
                           **kwargs):
    """
    """
    # https://github.com/biomedical-translation-corpora/corpora
    # git clone the repo to the WM_SAVE_DIR using subprocess
    assert f"33276392_de.txt" in os.listdir(WMT_SAVE_DIR), f"The biomed translation corpus project needs to be cloned to {WMT_SAVE_DIR}"
    # TODO: I only loaded part of the dataset so far (May 2nd), we should load the whole thing
    return True

def step_compare_dp_default_tokenization(dataset_path, 
                                        language_pair: str,
                                         model_tokenizer, 
                                         **kwargs) -> pl.DataFrame:
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

    # default_tokenization_strings = []
    dp_token_lengths = []
    default_token_lengths = []
    lang_codes = []
    for sample_txt_fname in tqdm(os.listdir(dataset_path)):
        with open(f"{dataset_path}/{sample_txt_fname}") as f:
            txt = f.read().strip()
            default_tokenizer_length = len(tokenizer.encode(txt))
            dp_encoded_text = dp_encode_bloom(txt)
            dp_tokenizer_length = len(dp_encoded_text)
            assert default_tokenizer_length >= dp_tokenizer_length, assert_action(f"DP tokenization is longer than default tokenization for {txt}")
            assert invert_dp_tokenize(dp_encoded_text) == txt, assert_action(f"DP tokenization is not reversible for {txt}")
        lang_code = sample_txt_fname.split("_")[1].split(".")[0]
        lang_codes.append(lang_code)
        dp_token_lengths.append(dp_tokenizer_length)
        default_token_lengths.append(default_tokenizer_length)
    # TODO: fill this out
    result_frame = pl.DataFrame({
        'dp_token_lengths': dp_token_lengths,
        'default_token_lengths': default_token_lengths,
        'lang_codes': lang_codes
    })
    ipdb.set_trace()
    return 


def step_train_model(
    model_name: str,
    dataset_path: str,
    language_pair: str,
    mapping_algorithm: str, 
    **kwargs
):
    # TODO
    tokenizer = get_tokenizer()
    def apply_tokenizer():
        def _tokenize(example):
            source = example["source"]
            target = example["target"]
            tokenized_dict = tokenizer(source)#, max_length=30, truncation=True)
            tokenized_dict["labels"] = tokenizer(target)["input_ids"]#, max_length=30, truncation=True)["input_ids"]
            # tokenized_dict['label'] = label_2_id[example[label_column][0]]
            return tokenized_dict
        return _tokenize
    
    filenames = set([os.basename(fn) for fn in tqdm(os.listdir(dataset_path))])
    SRC_LANG = "en"
    TGT_LANG = "de"
    sources = []
    targets = []
    for base_filename in tqdm(filenames):
        sample_txt_srcname = base_filename + f".{SRC_LANG}"
        sample_txt_tgtname = base_filename + f".{TGT_LANG}"
        with open(f"{dataset_path}/{sample_txt_srcname}") as f:
            sources.append(f.read().strip())
        with open(f"{dataset_path}/{sample_txt_tgtname}") as f:
            targets.append(f.read().strip())
    translation_df = pd.DataFrame({
        "source": sources,
        "target": targets,
    })
    translation_dataset = Dataset.from_pandas(translation_df)
    translation_dataset.map(apply_tokenizer)



if __name__ == '__main__':
    steps = OrderedDict()
    steps['download_biomed_dataset'] = SingletonStep(step_download_datasets, {
        'language_pair': 'en-de',
        'version': '001'
    })

    steps['compare_dp_default_tokenization'] = SingletonStep(step_compare_dp_default_tokenization, {
        'dataset_path': f"{WMT_SAVE_DIR}",
        'language_pair': 'en-de',
        'model_tokenizer': 'bigscience/bloom-3b', 
        'version': '001'
    })
    steps['train_model'] = SingletonStep(
        step_train_model, {
            'model_name': 'bigscience/bloomz-560m',
            'dataset_path': f"{WMT_SAVE_DIR}",
            'language_pair': 'en-de',
            'mapping_algorithm': 'default',
            'version': '001',
        }
    )
    # conduct()
    conduct(os.path.join(SCRATCH_DIR, "tokenization_cache"), steps, "tokenization_logs")
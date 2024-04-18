from dotenv import load_dotenv
import os
import polars as pl
import subprocess
from flowmason import SingletonStep, MapReduceStep
from collections import OrderedDict


load_dotenv()
WMT_SAVE_DIR = os.getenv("WMT_DATASET_SAVE_DIR") # create .env file in root and set this to whereever to save

def step_download_datasets(language_pair, 
                           **kwargs):
    """
    """
    # https://github.com/biomedical-translation-corpora/corpora
    # git clone the repo to the WM_SAVE_DIR using subprocess
    assert f"{WMT_SAVE_DIR}/biomedical-translation-corpora" in os.listdir(WMT_SAVE_DIR), f"The biomed translation corpus project needs to be cloned to {WMT_SAVE_DIR}"
    # call the python script in the shared task repository to download the datasets for {lang_pair}.
    pass
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
    pass

if __name__ == '__main__':
    steps = OrderedDict()
    steps['download_biomed_dataset'] = SingletonStep(step_download_datasets, {
        'language_pair': 'en-de'
        'version': '001'
    })

    steps['compare_dp_default_tokenization'] = SingletonStep(step_compare_dp_default_tokenization, {
        'dataset_path': f"{WMT_SAVE_DIR}/biomedical-translation-corpora/data/en-de/train.en",
        'language_pair': 'en-de',
        'model_tokenizer': 'bloomz/mbart-large-cc25'
    })

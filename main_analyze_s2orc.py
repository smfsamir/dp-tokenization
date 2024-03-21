import os
from transformers import AutoTokenizer, WhisperTokenizer
from flowmason import SingletonStep, MapReduceStep
from dotenv import load_dotenv
from collections import OrderedDict
from huggingface_hub import snapshot_download, login
from flowmason import conduct


from inspect_tokenizer import compute_shortest_tokenizations
from packages.constants import SCRATCH_DIR

load_dotenv()

def step_download_datasets(**kwargs): 
    # NOTE: might have to update this path if the snapshot changes
    os.makedirs(f"{SCRATCH_DIR}/tokenization", exist_ok=True)
    snapshot_download(repo_id = "leminda-ai/s2orc_small", cache_dir=f"{SCRATCH_DIR}/tokenization")
    return True # necessary so it doesn't keep downloading the dataset

def download_s2orc_corpus():
    return True

if __name__ == '__main__':
    cache_location = os.getenv("CACHE_DIR")
    steps = OrderedDict()
    cache_dir = SCRATCH_DIR + 'tokenization'
    conduct(cache_dir, steps, '')
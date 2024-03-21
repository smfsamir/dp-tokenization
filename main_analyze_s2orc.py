import ipdb
import os
from transformers import AutoTokenizer, WhisperTokenizer
from flowmason import SingletonStep, MapReduceStep
from dotenv import load_dotenv
from collections import OrderedDict
from huggingface_hub import snapshot_download, login
from flowmason import conduct
from datasets import load_dataset


from inspect_tokenizer import compute_shortest_tokenizations
from packages.constants import SCRATCH_DIR

load_dotenv()

def step_download_datasets(**kwargs): 
    # NOTE: might have to update this path if the snapshot changes
    snapshot_download(repo_id = "leminda-ai/s2orc_small", repo_type="dataset", cache_dir=SCRATCH_DIR)
    return True # necessary so it doesn't keep downloading the dataset

def step_iterate_dataset(**kwargs):
    # load the dataset
    dataset = load_dataset(path=os.path.join(SCRATCH_DIR, "datasets--leminda-ai--s2orc_small", 
                                             "snapshots", "6920172fe611a81f0a91e437a13c290a1ac114e7"))
    ipdb.set_trace()

if __name__ == '__main__':
    cache_location = os.getenv("CACHE_DIR")
    steps = OrderedDict()
    steps['download_datasets'] = SingletonStep(step_download_datasets, {
        'version': '001'
    })
    steps['step_iterate_dataset'] = SingletonStep(step_iterate_dataset, {
        'version': '001'
    })
    # steps['download_s2orc_corpus'] = SingletonStep(download_s2orc_corpus, {
    #     'version': '001'
    # }, cache_location)
    conduct(os.path.join(SCRATCH_DIR, "tokenization_cache"), steps, "tokenization_logs")
import ipdb
import os
from transformers import AutoTokenizer, WhisperTokenizer, AutoModelForCausalLM
from flowmason import SingletonStep, MapReduceStep
from dotenv import load_dotenv
from collections import OrderedDict
from huggingface_hub import snapshot_download, login
from flowmason import conduct
from datasets import load_dataset
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq
from typing import Dict, Tuple, List
import torch
from dataclasses import dataclass

from inspect_tokenizer import compute_shortest_tokenizations
from packages.constants import SCRATCH_DIR

load_dotenv()


@dataclass
class DataCollatorCustomTokenization:
    tokenizer: AutoTokenizer

    # def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
    def __call__(self, batch: Dict[str, str]) -> Dict[str, torch.Tensor]:
        snippets = [batch[i]['paperAbstract'] for i in range(len(batch))]
        topics = [batch[i]['fieldsOfStudy'] for i in range(len(batch))]
        ipdb.set_trace()

        batch['input_ids'] = self.tokenizer(snippets, truncation=True, padding=True, max_length=2048, return_tensors="pt")["input_ids"]
        labels = self.tokenizer(topics, truncation=True, padding=True, max_length=2048, return_tensors="pt")["input_ids"]
        batch["labels"] = labels
        return batch



def step_download_datasets(**kwargs): 
    # NOTE: might have to update this path if the snapshot changes
    snapshot_download(repo_id = "leminda-ai/s2orc_small", repo_type="dataset", cache_dir=SCRATCH_DIR)
    return True # necessary so it doesn't keep downloading the dataset

def step_download_olmo_model(**kwargs):
    access_token = os.getenv("HF_ACCESS_TOKEN")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir=SCRATCH_DIR, token=access_token)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir=SCRATCH_DIR, token=access_token)
    return True

def step_iterate_dataset(**kwargs):
    # load the dataset
    # ipdb.set_trace()
    pass

def step_finetune_llama(**kwargs):
    # load the first 10 percent as eval dataset
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir=SCRATCH_DIR)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir=SCRATCH_DIR)
    eval_dataset = load_dataset("leminda-ai/s2orc_small", split='train[:10%]', cache_dir=SCRATCH_DIR)
    train_dataset = load_dataset("leminda-ai/s2orc_small", split='train[10%:]', cache_dir=SCRATCH_DIR)
    collator = DataCollatorCustomTokenization(tokenizer)
    training_args = TrainingArguments(
        output_dir=f"{SCRATCH_DIR}/llama_7b_hf_finetuned",
        learning_rate=1e-3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=2,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        remove_unused_columns=False
    )
    ipdb.set_trace()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=collator
    )
    trainer.train()



if __name__ == '__main__':
    cache_location = os.getenv("CACHE_DIR")
    steps = OrderedDict()
    steps['download_datasets'] = SingletonStep(step_download_datasets, {
        'version': '001'
    })
    steps['step_download_olmo_model'] = SingletonStep(step_download_olmo_model, {
        'version': '001'
    })
    steps['step_iterate_dataset'] = SingletonStep(step_iterate_dataset, {
        'version': '001'
    })
    steps['step_finetune_llama'] = SingletonStep(step_finetune_llama, {
        'version': '001'
    })
    # steps['download_s2orc_corpus'] = SingletonStep(download_s2orc_corpus, {
    #     'version': '001'
    # }, cache_location)
    conduct(os.path.join(SCRATCH_DIR, "tokenization_cache"), steps, "tokenization_logs")
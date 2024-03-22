import ipdb
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from flowmason import SingletonStep, MapReduceStep
from peft import LoraConfig, TaskType, get_peft_model
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
    def __call__(self, input_batch: Dict[str, str]) -> Dict[str, torch.Tensor]:
        snippets = [input_batch[i]['paperAbstract'] for i in range(len(input_batch))]

        # topics = [' '.join(batch[i]['fieldsOfStudy']) for i in range(len(batch))]
        # write as a for loop for now
        topics = []
        for i in range(len(input_batch)):
            topics.append(' '.join(input_batch[i]['fieldsOfStudy']))

        output_batch = self.tokenizer(snippets, truncation=True, padding=True, max_length=2048, return_tensors="pt")

        labels = self.tokenizer(topics, truncation=True, padding=True, max_length=2048, return_tensors="pt")
        labels = labels["input_ids"].masked_fill(labels.attention_mask.ne(1), -100)

        output_batch["labels"] = labels
        return output_batch


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
    compute_dtype = getattr(torch, "float16")
    # quant_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=compute_dtype,
    #     bnb_4bit_use_double_quant=False,
    # )
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir=SCRATCH_DIR)
    peft_config = LoraConfig(task_type = TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
    model = get_peft_model(model, peft_config)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir=SCRATCH_DIR)
    eval_dataset = load_dataset("leminda-ai/s2orc_small", split='train[:10%]', cache_dir=SCRATCH_DIR)
    train_dataset = load_dataset("leminda-ai/s2orc_small", split='train[10%:]', cache_dir=SCRATCH_DIR)
    collator = DataCollatorCustomTokenization(tokenizer)
    tokenizer.pad_token = "[PAD]"
    tokenizer.padding_side = "right"

    training_args = TrainingArguments(
        output_dir=f"{SCRATCH_DIR}/llama_7b_hf_finetuned",
        learning_rate=1e-4,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=2,
        weight_decay=0.01,
        logging_steps=25,
        max_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        remove_unused_columns=False
    )
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
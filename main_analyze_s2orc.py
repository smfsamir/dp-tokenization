import numpy as np
import loguru
from functools import partial
import ipdb
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoModelForSequenceClassification, DataCollatorWithPadding
import evaluate
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
logger = loguru.logger


# @dataclass
# class DataCollatorCustomTokenization:
#     tokenizer: AutoTokenizer
#     label2id: Dict[str, int]

#     # def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
#     def __call__(self, input_batch: Dict[str, str]) -> Dict[str, torch.Tensor]:
#         snippets = [input_batch[i]['paperAbstract'] for i in range(len(input_batch))]

#         # topics = [' '.join(batch[i]['fieldsOfStudy']) for i in range(len(batch))]
#         # write as a for loop for now
#         topics = []
#         for i in range(len(input_batch)):
#             topics.append(', '.join(input_batch[i]['fieldsOfStudy']))
        
#         # combine the snippets and topics with a colon
#         complete_sentences = [f"{snippets[i]}: {topics[i]}" for i in range(len(snippets))]
#         output_batch = self.tokenizer(complete_sentences, truncation=True, padding=True, max_length=2048, return_tensors="pt")
#         output_batch["labels"] = output_batch["input_ids"].clone().masked_fill(output_batch.attention_mask.ne(1), -100)
#         return output_batch


def step_download_datasets(**kwargs): 
    # NOTE: might have to update this path if the snapshot changes
    snapshot_download(repo_id = "leminda-ai/s2orc_small", repo_type="dataset", cache_dir=SCRATCH_DIR)
    return True # necessary so it doesn't keep downloading the dataset

def step_download_olmo_model(**kwargs):
    access_token = os.getenv("HF_ACCESS_TOKEN")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir=SCRATCH_DIR, token=access_token)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir=SCRATCH_DIR, token=access_token)
    # TODO: add the label to preprocessing and then commence training.
    return True


def llama_preprocessing_function(llama_tokenizer, label_2_id):
    label_column = "fieldsOfStudy"
    def _tokenize(example):
        tokenized_dict = llama_tokenizer(example['paperAbstract'], truncation=True, max_length=2048)
        tokenized_dict['label'] = label_2_id[example[label_column][0]]
        return tokenized_dict
    return _tokenize

def step_iterate_dataset(**kwargs):
    # load the dataset
    # ipdb.set_trace()
    pass

def compute_metrics(eval_pred):
    # All metrics are already predefined in the HF `evaluate` package
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric= evaluate.load("f1")
    accuracy_metric = evaluate.load("accuracy")

    logits, labels = eval_pred # eval_pred is the tuple of predictions and labels returned by the model
    predictions = np.argmax(logits, axis=-1)
    precision = precision_metric.compute(predictions=predictions, references=labels, average='macro')["precision"]
    recall = recall_metric.compute(predictions=predictions, references=labels, average='macro')["recall"]
    f1 = f1_metric.compute(predictions=predictions, references=labels, average='macro')["f1"]
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels, average='macro')["accuracy"]
    # The trainer is expecting a dictionary where the keys are the metrics names and the values are the scores. 
    return {"precision": precision, "recall": recall, "f1-score": f1, 'accuracy': accuracy}


def step_finetune_llama(**kwargs):
    # load the first 10 percent as eval dataset
    compute_dtype = getattr(torch, "bfloat16")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )
    # model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir=SCRATCH_DIR, quantization_config=quant_config, torch_dtype=compute_dtype)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir=SCRATCH_DIR)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.pad_token = "[PAD]"

    logger.info("Loading the dataset")
    remove_columns =  ['id', 'title', 'paperAbstract', 'entities', 's2Url', 'pdfUrls', 's2PdfUrl', 'authors', 
                       'inCitations', 'outCitations', 'fieldsOfStudy', 'year', 'venue', 
                       'journalName', 'journalVolume', 'journalPages', 'sources', 
                       'doi', 'doiUrl', 'pmid', 'magId']
    eval_dataset = load_dataset("leminda-ai/s2orc_small", split='train[:5%]', cache_dir=SCRATCH_DIR).filter(lambda x: len(x['fieldsOfStudy']) == 1).select(range(1000))
    logger.info(f"Loaded evaluation dataset; {len(eval_dataset)} examples")
    train_dataset = load_dataset("leminda-ai/s2orc_small", split='train[5%:10%]', cache_dir=SCRATCH_DIR).filter(lambda x: len(x['fieldsOfStudy']) == 1)
    logger.info(f"Loaded training dataset; {len(train_dataset)} examples")
    # preprocess the dataset by tokenizing the text

    unique_fields = list(set([field for example in eval_dataset for field in example['fieldsOfStudy']]))
    num_fields = len(unique_fields)
    # id2label = {i: field for i, field in enumerate(unique_fields)}
    label2id = {field: i for i, field in enumerate(unique_fields)}
    print(f"The unique fields are {unique_fields}")
    llama_preprocess = llama_preprocessing_function(tokenizer, label2id)
    logger.info("Preprocessing the dataset")
    eval_dataset = eval_dataset.map(llama_preprocess, remove_columns=remove_columns)
    train_dataset = train_dataset.map(llama_preprocess, remove_columns=remove_columns)
    logger.info("Preprocessed the dataset")
    logger.info("Loading the model")
    model = AutoModelForSequenceClassification.from_pretrained("meta-llama/Llama-2-7b-hf", 
                                                               cache_dir=SCRATCH_DIR, 
                                                               quantization_config=quant_config, 
                                                               torch_dtype=compute_dtype, 
                                                                num_labels=num_fields 
                                                                )
    logger.info("Loaded the model")
    model.config.pad_token_id = model.config.eos_token_id
    peft_config = LoraConfig(task_type = TaskType.SEQ_CLS, inference_mode=False, r=16, lora_alpha=16, lora_dropout=0.05, bias="none", 
                             target_modules=[
                                    "q_proj",
                                    "v_proj",  
                                ],
                            )
    model = get_peft_model(model, peft_config)
    llama_data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    # tokenizer.padding_side = "right"

    training_args = TrainingArguments(
        output_dir=f"{SCRATCH_DIR}/llama_7b_hf_finetuned_lora",
        learning_rate=1e-4,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        optim="paged_adamw_32bit",
        gradient_accumulation_steps=4,
        num_train_epochs=2,
        weight_decay=0.01,
        logging_steps=25,
        max_steps=100,
        max_grad_norm=0.3,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        remove_unused_columns=False
    )
    logger.info(f"Starting the trainer")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=llama_data_collator, 
        compute_metrics=compute_metrics
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
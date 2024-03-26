import polars as pl
from tqdm import tqdm
import numpy as np
import loguru
from functools import partial
import ipdb
import os
from peft import AutoPeftModelForSequenceClassification, PeftModel
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
from bidict import bidict

from inspect_tokenizer import compute_shortest_tokenizations
from packages.tokenizer_utils import dp_tokenize_llama, pretokenize_with_llama
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

def llama_preprocessing_function(llama_tokenizer, tokenize_method, label_2_id):
    label_column = "fieldsOfStudy"
    tokenize_func = None
    assert tokenize_method in ["default", "dp"], logger.error(f"Tokenize method {tokenize_method} not recognized")
    if tokenize_method == "default":
        logger.info("Using the default tokenizer to tokenize the text")
        def _tokenize(example):
            tokenized_dict = llama_tokenizer(example['paperAbstract'], truncation=True, max_length=2048)
            tokenized_dict['label'] = label_2_id[example[label_column][0]]
            return tokenized_dict
        tokenize_func = _tokenize
    elif tokenize_method == "dp":
        logger.info("Using dynamic programming to tokenize the text")
        dp_tokenize, decode_dp_tokenization = dp_tokenize_llama(llama_tokenizer)
        def _tokenize(example):
            tokenized_dict = {}
            try:
                shortest_tokenization = dp_tokenize(example['paperAbstract'])
            except:
                logger.error(f"Failed to tokenize the text {example['paperAbstract']}")
                return None
            
            # assert decode_dp_tokenization(shortest_tokenization) == example['paperAbstract'], ipdb.set_trace()
            try:
                assert decode_dp_tokenization(shortest_tokenization) == example['paperAbstract']
            except:
                logger.error(f"Failed to decode the tokenization {shortest_tokenization}\n\n for the text {example['paperAbstract']}")
            tokenized_dict['input_ids'] = shortest_tokenization
            tokenized_dict['attention_mask'] = [1] * len(shortest_tokenization)
            tokenized_dict['label'] = label_2_id[example[label_column][0]]
            return tokenized_dict
        tokenize_func = _tokenize
    return tokenize_func

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
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
    # The trainer is expecting a dictionary where the keys are the metrics names and the values are the scores. 
    return {"precision": precision, "recall": recall, "f1-score": f1, 'accuracy': accuracy}

def load_base_model(num_fields: int):
    compute_dtype = getattr(torch, "bfloat16")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )
    model = AutoModelForSequenceClassification.from_pretrained("meta-llama/Llama-2-7b-hf", 
                                                               cache_dir=SCRATCH_DIR, 
                                                               quantization_config=quant_config, 
                                                               torch_dtype=compute_dtype, 
                                                                num_labels=num_fields 
                                                                )
    return model

def step_select_train_indices(**kwargs):
    train_dataset = load_dataset("leminda-ai/s2orc_small", split='train[5%:50%]', cache_dir=SCRATCH_DIR).filter(lambda x: len(x['fieldsOfStudy']) == 1)
    ipdb.set_trace()

def step_finetune_llama(tokenize_method, **kwargs):
    """Finetune the Llama model on the S2ORC dataset.

    tokenize_method (str): one of ['default', 'dp', 'flota']
    """
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
    train_dataset = load_dataset("leminda-ai/s2orc_small", split='train[5%:50%]', cache_dir=SCRATCH_DIR).filter(lambda x: len(x['fieldsOfStudy']) == 1)
    logger.info(f"Loaded training dataset; {len(train_dataset)} examples")
    ipdb.set_trace()
    # preprocess the dataset by tokenizing the text

    unique_fields = list(set([field for example in eval_dataset for field in example['fieldsOfStudy']]))
    num_fields = len(unique_fields)
    # id2label = {i: field for i, field in enumerate(unique_fields)}
    label2id = {'Psychology': 0, 'Geography': 1, 'Geology': 2, 'Art': 3, 'Engineering': 4, 'Philosophy': 5, 'Medicine': 6, 'Sociology': 7, 'History': 8, 'Computer Science': 9, 'Physics': 10, 'Political Science': 11, 'Chemistry': 12, 'Environmental Science': 13, 'Materials Science': 14, 'Mathematics': 15, 'Economics': 16, 'Biology': 17, 'Business': 18}
    logger.info(f"The mapping from a label to an id is {label2id}")
    print(f"The unique fields are {unique_fields}")
    llama_preprocess = llama_preprocessing_function(tokenizer, tokenize_method, label2id)
    logger.info("Preprocessing the dataset")
    eval_dataset = eval_dataset.map(llama_preprocess, remove_columns=remove_columns)
    train_dataset = train_dataset.map(llama_preprocess, remove_columns=remove_columns)
    model = load_base_model(num_fields)
    logger.info("Preprocessed the dataset")
    logger.info("Loading the model")
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
        output_dir=f"{SCRATCH_DIR}/llama_7b_hf_finetuned_lora_{tokenize_method}",
        per_device_train_batch_size=1,
        lr_scheduler_type="linear",
        learning_rate=1e-5,
        per_device_eval_batch_size=1,
        optim="paged_adamw_32bit",
        gradient_accumulation_steps=4,
        num_train_epochs=2,
        weight_decay=0.01,
        logging_steps=100,
        warmup_ratio=0.1, 
        max_steps=4000,
        eval_steps = 1000,
        save_steps = 1000,
        max_grad_norm=0.3,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        remove_unused_columns=False,
        save_total_limit=2
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
    model.save_pretrained(f"{SCRATCH_DIR}/llama_7b_hf_finetuned_lora")
    logger.info(f"Finished training the model; saved it to {SCRATCH_DIR}/llama_7b_hf_finetuned_lora")

def step_login(**kwargs):
    access_token = os.environ["HF_ACCESS_TOKEN"]
    login(token=access_token)

def step_probe_eval_dataset(**kwargs):
    llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir=SCRATCH_DIR)
    eval_dataset = load_dataset("leminda-ai/s2orc_small", split='train[:5%]', cache_dir=SCRATCH_DIR).filter(lambda x: len(x['fieldsOfStudy']) == 1)
    eval_abstracts = [example['paperAbstract'] for example in eval_dataset]
    eval_domains = [example['fieldsOfStudy'][0] for example in eval_dataset]
    dp_tokenize, decode_dp_tokenization = dp_tokenize_llama(llama_tokenizer)

    dp_lengths = []
    default_lengths = []
    progress = tqdm(total=len(eval_abstracts))
    total_improved = 0
    total = 0 

    vocab = bidict(llama_tokenizer.get_vocab())
    pretokenize_func = pretokenize_with_llama(llama_tokenizer, vocab)

    improved_tokens = [] # List[List[str]]
    worse_tokens = [] # List[List[str]]
    for i in range(len(eval_domains)):
        abstract = eval_abstracts[i]
        dp_length = len(dp_tokenize(abstract))
        default_length = len(llama_tokenizer.encode(abstract))
        dp_lengths.append(dp_length)
        default_lengths.append(default_length)
        if dp_length < default_length:
            total_improved += 1
            # TODO: pretokenize the abstract, and find exactly which tokens get shortened. 
            abstract_pretok = pretokenize_func(abstract)
            improved_tokens_specific = []
            worse_tokens_specific = []
            for token in abstract_pretok:
                if len(dp_tokenize(token)) < len(llama_tokenizer.encode(token)):
                    improved_tokens_specific.append(
                        [vocab.inverse[token] for token in dp_tokenize(token)]
                    )
                    worse_tokens_specific.append(
                        [vocab.inverse[token] for token in llama_tokenizer.encode(token)]
                    )
            improved_tokens.append(improved_tokens_specific)
            worse_tokens.append(worse_tokens_specific)
        else:
            improved_tokens.append([])
            worse_tokens.append([])

        total += 1
        if i % 1000 == 0:
            logger.info(f"{total_improved}/{total} examples are shorter")
        progress.update(1)
    result_frame = pl.DataFrame({
        "abstract": eval_abstracts,
        "domain": eval_domains,
        "dp_length": dp_lengths,
        "default_length": default_lengths, 
        "improved_tokens": improved_tokens,
        "worse_tokens": worse_tokens
    }) 
    result_frame.write_json("dp_vs_default_tokenization_s2orc.json")
    ipdb.set_trace()

def step_load_trained_model(trained_checkpoint_path, **kwargs):
    # model = AutoPeftModelForSequenceClassification.from_pretrained(trained_checkpoint_path)
    model = load_base_model(19)
    model = PeftModel.from_pretrained(model, trained_checkpoint_path)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir=SCRATCH_DIR)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer('General thermodynamic relations for the work of polydisperse micelle formation in the model of ideal solution of molecular aggregates in nonionic surfactant solution and the model of "dressed micelles" in ionic solution have been considered. In particular, the dependence of the aggregation work on the total concentration of nonionic surfactant has been analyzed. The analogous dependence for the work of formation of ionic aggregates has been examined with regard to existence of two variables of a state of an ionic aggregate, the aggregation numbers of surface active ions and counterions. To verify the thermodynamic models, the molecular dynamics simulations of micellization in nonionic and ionic surfactant solutions at two total surfactant concentrations have been performed. It was shown that for nonionic surfactants, even at relatively high total surfactant concentrations, the shape and behavior of the work of polydisperse micelle formation found within the model of the ideal solution at different total surfactant concentrations agrees fairly well with the numerical experiment. For ionic surfactant solutions, the numerical results indicate a strong screening of ionic aggregates by the bound counterions. This fact as well as independence of the coefficient in the law of mass action for ionic aggregates on total surfactant concentration and predictable behavior of the "waterfall" lines of surfaces of the aggregation work upholds the model of "dressed" ionic aggregates.', return_tensors="pt", padding=True, truncation=True, max_length=2048)

    outputs = model(**inputs)
    logits = outputs.logits
    label2id = {'Psychology': 0, 'Geography': 1, 'Geology': 2, 'Art': 3, 'Engineering': 4, 'Philosophy': 5, 'Medicine': 6, 'Sociology': 7, 'History': 8, 'Computer Science': 9, 'Physics': 10, 'Political Science': 11, 'Chemistry': 12, 'Environmental Science': 13, 'Materials Science': 14, 'Mathematics': 15, 'Economics': 16, 'Biology': 17, 'Business': 18}
    id2label = {v: k for k, v in label2id.items()}
    prediction = torch.argmax(logits, dim=-1)
    logger.info(f"The prediction is {id2label[prediction]}")
    # tokenizer = AutoTokenizer.from_pretrained(trained_checkpoint_path)


if __name__ == '__main__':
    cache_location = os.getenv("CACHE_DIR")
    steps = OrderedDict()
    steps['download_datasets'] = SingletonStep(step_download_datasets, {
        'version': '001'
    })
    steps['step_login'] = SingletonStep(step_login, {
        'version': '001'
    })
    steps['step_iterate_dataset'] = SingletonStep(step_iterate_dataset, {
        'version': '001'
    })
    # steps['step_finetune_llama_default'] = SingletonStep(step_finetune_llama, {
    #     'tokenize_method': 'default',
    #     'version': '001'
    # })
    # steps['step_finetune_llama_dp'] = SingletonStep(step_finetune_llama, {
    #     'tokenize_method': 'dp',
    #     'version': '001'
    # })
    steps['step_probe_eval_dataset'] = SingletonStep(step_probe_eval_dataset, {
        'version': '001'
    })
    # steps['step_inspect_finedtuned_llama'] = SingletonStep(step_load_trained_model,
    # {
    #     'version': '001', 
    #     'trained_checkpoint_path': f"{SCRATCH_DIR}/llama_7b_hf_finetuned_lora"
    # })
    conduct(os.path.join(SCRATCH_DIR, "tokenization_cache"), steps, "tokenization_logs")
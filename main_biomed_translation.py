import os
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
import transformers
# import DataCollatorForSeq2Seq
from transformers import DataCollatorForLanguageModeling
from transformers import AutoTokenizer, WhisperTokenizer, AutoModelForCausalLM
import evaluate
import numpy as np
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


def get_tokenizer(default_tokenizer, mapping_algorithm):
    if mapping_algorithm == "dp":
        # TODO: Add an assertion to be sure inverting recovers the original text.
        dp_encode_bloom, invert_dp_tokenize = dp_tokenize_bloom(
            default_tokenizer, HF_CACHE_DIR
        )
        return dp_encode_bloom
    else:
        return default_tokenizer.encode


class ShortcutDataCollatorForSeq2Seq(DataCollatorForLanguageModeling):

    def __call__(self, features, return_tensors=None):
        print(features)
        padded_features = super().__call__(features, return_tensors)
        print(padded_features)
        raise Exception("Check that the features and padded features match (modulo the padding)")
        return padded_features


def step_train_model(
    model_name: str,
    dataset_path: str,
    language_pair: str,
    mapping_algorithm: str,
    output_dir: str,
    **kwargs
):
    # Get the base tokenizer
    default_tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=HF_CACHE_DIR)
    tokenizer = get_tokenizer(default_tokenizer, mapping_algorithm)

    def apply_tokenizer(example):
        source = example["source"]
        target = example["target"]
        tokenized_dict = {
            'input_ids': tokenizer(source),
            'labels': tokenizer(target)
        }
        return tokenized_dict
    
    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]
        return preds, labels

    metric = evaluate.load("sacrebleu")
    def _compute_metrics(eval_preds):
        preds, labels = eval_preds
        print("Eval Sample prediction and labels:")
        print(preds[:2],labels[:2])
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds != -100, preds, default_tokenizer.pad_token_id)
        decoded_preds = default_tokenizer.batch_decode(preds, skip_special_tokens=True)
        print(decoded_preds[:2])

        labels = np.where(labels != -100, labels, default_tokenizer.pad_token_id)
        decoded_labels = default_tokenizer.batch_decode(labels, skip_special_tokens=True)
        print(decoded_labels[:2])
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": result["score"]}

        prediction_lens = [np.count_nonzero(pred != default_tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result
    
    filenames = set([os.path.basename(fn).split('_')[0] for fn in tqdm(os.listdir(dataset_path))])
    SRC_LANG = "en"
    TGT_LANG = "de"
    sources = []
    targets = []

    for base_filename in tqdm(filenames):
        sample_txt_srcname = base_filename + f"_{SRC_LANG}.txt"
        sample_txt_tgtname = base_filename + f"_{TGT_LANG}.txt"
        with open(f"{dataset_path}/{sample_txt_srcname}") as f:
            sources.append(f.read().strip())
        with open(f"{dataset_path}/{sample_txt_tgtname}") as f:
            targets.append(f.read().strip())
    translation_df = pd.DataFrame({
        "source": sources,
        "target": targets,
    })
    translation_dataset = Dataset.from_pandas(translation_df)
    translation_dataset = translation_dataset.map(apply_tokenizer)
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=HF_CACHE_DIR)
    training_args = transformers.Seq2SeqTrainingArguments(
        report_to="wandb",
        run_name=f"sanity_check-{SRC_LANG}-{TGT_LANG}",
        output_dir=f"{output_dir}/bloom_dp_560m",
        do_eval=True,
        logging_steps=50,
        evaluation_strategy="steps",
        eval_steps=20,
        learning_rate=0.001,
        warmup_steps=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=0,
        save_total_limit=3,
        num_train_epochs=10,
        predict_with_generate=True,
        generation_max_length=128,
    )
    data_collator = DataCollatorForLanguageModeling(
        default_tokenizer, padding="max_length", max_length=1024
    )
    trainer = transformers.Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=translation_dataset,
        eval_dataset=translation_dataset,
        #tokenizer=tokenizer, #TODO: what interface does tokenizer need to implement?
        data_collator=data_collator,
        compute_metrics=_compute_metrics,
    )
    print(trainer)
    print("Training")
    print("Parallel mode: ", trainer.args.parallel_mode)
    trainer.train()


if __name__ == '__main__':
    steps = OrderedDict()
    steps['download_biomed_dataset'] = SingletonStep(step_download_datasets, {
        'language_pair': 'en-de',
        'version': '001'
    })

    # steps['compare_dp_default_tokenization'] = SingletonStep(step_compare_dp_default_tokenization, {
    #     'dataset_path': f"{WMT_SAVE_DIR}",
    #     'language_pair': 'en-de',
    #     'model_tokenizer': 'bigscience/bloom-3b', 
    #     'version': '001'
    # })
    steps['train_model'] = SingletonStep(
        step_train_model, {
            'model_name': 'bigscience/bloomz-560m',
            'dataset_path': f"{WMT_SAVE_DIR}",
            'language_pair': 'en-de',
            'mapping_algorithm': 'default',
            'output_dir': SCRATCH_DIR,
            'version': '001',
        }
    )
    # conduct()
    conduct(os.path.join(SCRATCH_DIR, "tokenization_cache"), steps, "tokenization_logs")
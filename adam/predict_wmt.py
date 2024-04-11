import transformers
from transformers.data.data_collator import DataCollatorForSeq2Seq
from datasets import load_dataset
import torch
import evaluate
import numpy as np

import argparse
from dataclasses import dataclass
from typing import Set

from lib.flota.src.flota import FlotaTokenizer
import optimal_tokenizer


# For eval.
metric = evaluate.load("sacrebleu")

# Hyerparams
BATCH_SIZE = 8

SOURCE_LANG="en"
TARGET_LANG="de"


def get_model(model_name: str, checkpoint: str):
    model_factory = {
        "google/mt5-small": (
            transformers.AutoTokenizer.from_pretrained(
                "google/mt5-small",
                legacy=False,
            ),
            transformers.AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
        ),
        #"bigscience/bloom-3b": (
        #    transformers.AutoTokenizer.from_pretrained("bigscience/bloom-3b"),
        #    transformers.AutoModelForConditionalGeneration.from_pretrained("bigscience/bloom-3b")
        #)
    }
    if model_name in model_name:
        return model_factory[model_name]
    else:
        msg = f"No model_name {model_name}"
        raise NotImplementedError(msg)


def get_post_tokenizer(
    name: str,
    vocabulary: Set[str],
    model_name: str,
    tokenizer: transformers.PreTrainedTokenizer
):
    post_tokenizer_fac = {
        # TODO: What should k be? In the paper I think they found 3-4 was good?
        "flota": lambda: FlotaTokenizer(model_name, k=100, strict=False, mode="flota"),
        "opt": lambda: optimal_tokenizer.OptTokenizer(
            vocab=vocabulary,
            model_name=model_name,
            tokenizer=tokenizer
        ),
        "default": lambda: Dummy(),
    }
    if name in post_tokenizer_fac:
        return post_tokenizer_fac[name]()
    else:
        msg = f"No post_tokenizer_name {name}"
        raise NotImplementedError(msg)


def model_preprocessing_function(tokenizer):
    prefix = f"translate {SOURCE_LANG} to {TARGET_LANG}: "
    def _tokenize(example):
        source = prefix + example["translation"][SOURCE_LANG]
        trgt = example["translation"][TARGET_LANG]
        tokenized_dict = tokenizer(source)
        tokenized_dict["labels"] = tokenizer(trgt)["input_ids"]
        # tokenized_dict['label'] = label_2_id[example[label_column][0]]
        return tokenized_dict
    return _tokenize


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--post_tokenizer")
    args = parser.parse_args()
    # vocabulary = set(
    #     k.lstrip(special) for k, v in tokenizer.vocab.items()
    # )
    # TODO: Load vocabulary with special chars and implement method for handling them
    #       in the tokenizer.
    model_tokenizer, model = get_model(args.model_name, args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device, torch.cuda.is_available())
    model = model.to(device)
    print("Getting base tokenizer vocab")
    vocabulary = set(model_tokenizer.vocab.keys())
    print(f"Loaded {len(vocabulary)} vocab items. Adding all characters.")
    vocabulary = vocabulary.union(*[set(t) for t in vocabulary])
    print(f"Final vocab size: {len(vocabulary)}")
    # e.g. flota, optimal tokenization (ours)
    print(f"Getting {args.post_tokenizer} tokenizer")
    if args.post_tokenizer == "default":
        tokenizer = model_tokenizer
    else:
        tokenizer = get_post_tokenizer(
            args.post_tokenizer, vocabulary, args.model_name, model_tokenizer
        )
    # Get the de/en translation data
    # TODO: subsample to ~1/10
    print("Getting datasets")
    # train_dataset = load_dataset("iwslt2017", "iwslt2017-en-de", split="train")
    # for d in train_dataset:
    #     print(d)
    #     break
    eval_dataset = load_dataset("iwslt2017", "iwslt2017-en-de", split="validation")
    # for v in validation_dataset:
    #     print(v)
    #     break
    # print("Preprocessing...")
    # raise Exception("stop")
    model_preprocess = model_preprocessing_function(tokenizer)
    print("Preprocessing data")
    eval_dataset = eval_dataset.map(model_preprocess)#, remove_columns=remove_columns)
    print("Eval dataset preprocessed. Example: ")
    print(eval_dataset[0])
    # train_dataset = train_dataset.map(model_preprocess)#, remove_columns=remove_columns)
    # print("Train dataset preprocessed. Example: ")
    # print(train_dataset[0])
    def _compute_metrics(eval_preds):
        preds, labels = eval_preds
        print("Eval Sample prediction and labels:")
        print(preds[:2],labels[:2])
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds != -100, preds, model_tokenizer.pad_token_id)
        decoded_preds = model_tokenizer.batch_decode(preds, skip_special_tokens=True)
        print(decoded_preds[:2])

        labels = np.where(labels != -100, labels, model_tokenizer.pad_token_id)
        decoded_labels = model_tokenizer.batch_decode(labels, skip_special_tokens=True)
        print(decoded_labels[:2])
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": result["score"]}

        prediction_lens = [np.count_nonzero(pred != model_tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    data_collator = DataCollatorForSeq2Seq(model_tokenizer, padding="max_length", max_length=1024)
    training_args = transformers.Seq2SeqTrainingArguments(
        do_eval=True,
        per_device_eval_batch_size=BATCH_SIZE,
        predict_with_generate=True,
        generation_max_length=128,
    )
    trainer = transformers.Seq2SeqTrainer(
        args=training_args,
        model=model,
        data_collator=data_collator,
        compute_metrics=_compute_metrics,
    )
    print(trainer)
    print("Predicting")
    print("Parallel mode: ", trainer.args.parallel_mode)
    preds_tuple = trainer.predict(eval_dataset)
    print(preds_tuple["metrics"])

    # FIXME: We should also write this in a way that separates subword tokens for measuring predicted compressions.
    with open(f"{args.model_name}-{args.post_tokenizer}", "w") as out:
        for preds, labels in zip(preds_tuple["predictions"], preds_tuple["label_ids"]):
            preds = np.where(preds != -100, preds, model_tokenizer.pad_token_id)
            decoded_preds = model_tokenizer.batch_decode(preds, skip_special_tokens=True)

            labels = np.where(labels != -100, labels, model_tokenizer.pad_token_id)
            decoded_labels = model_tokenizer.batch_decode(labels, skip_special_tokens=True)
            decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
            for p, l in zip(decoded_preds, decoded_labels):
                print(p, l, sep="\t", file=out)


if __name__ == "__main__":
    main()

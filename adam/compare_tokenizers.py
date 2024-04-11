from typing import Dict, List, Set, Tuple
import re
import argparse

from tqdm import tqdm
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from datasets import load_dataset

from lib.flota.src.flota import FlotaTokenizer
from optimal_tokenizer import OptTokenizer


def read_flota_data(filename: str):
    return pd.DataFrame(filename)


def read_celex_data(filename: str, vocab: Set[str]) -> List[str]:
    """Reads the words I have subsampled from CELEX.

    Logs an analysis of how many full words are in the vocab, and perhaps some analysis
    of segments.

    Args:
        filename (str): The data file of subsampled words for a language.
        vocab (Set[str]): The vocab that we will use for tokenization.

    Returns:
        List[str]: words for tokenization.
    """
    df = pd.read_csv(filename, sep="\t")
    df = df.loc[~df["Head"].str.contains("-")]
    return df["Head"].tolist()


def read_ladec_data(filename: str, vocab: Set[str]) -> List[str]:
    """Reads the LADEC words.

    Returns:
        List[str]: words for tokenization.
    """
    df = pd.read_csv(filename)
    df = df.loc[~df["stim"].str.contains("-")]
    return df["stim"].tolist()


def get_best_path(node: Tuple, backpointers: Dict):
    prev_nodes = backpointers[node]
    # If length 1, it is a terminal.
    if len(prev_nodes) == 1:
        # i, j = prev_nodes
        return prev_nodes
    path = get_best_path(prev_nodes[0], backpointers)
    path += get_best_path(prev_nodes[1], backpointers)

    return path


def compute_length_of_most_efficient_tokenization(sequence: List[str], vocabulary: Set[str]):
    """Compute the least number of tokens required to represent {sequence} using tokens from
    the {vocabulary}. For example, if {sequence} is *in* the vocabulary, then the most efficient
    tokenization has a length of 1. 

    Args:
        sequence (List[str]): the sequence represented as a list of characters. E.g., for a sequence like "the weather", it will be provided as an input to this function as 
        ['t', 'h', 'e', ' ', 'w', 'e', 'a', 't', 'h', 'e', 'r'].  
        You can assume each of the individual characters are in the vocabulary (e.g., s[i], for i in range(len(sequence))).
        However, s[i:j] for j>i is not guaranteed to be in the vocabulary. 
        vocabulary: All of the tokens in the model's vocabulary. For HuggingFace models, obtaining this vocabulary is easy.
        For example, for Bloom-3B, have a look at tokenizer.json: https://huggingface.co/bigscience/bloom-3b/tree/main. 
    """
    # fill opt with np.inf
    opt = np.full((len(sequence), len(sequence)), np.inf)

    # set the diagonal to 1
    for i in range(len(sequence)):
        opt[i,i] = 1
        assert sequence[i] in vocabulary, f"{sequence[i]} not in vocabulary"

    # opt[i,j] = min(opt[i,k] + opt[k+1,j] for k in range(i,j) for k in i<=k<j, 1 if seq[i,j] in vocab) 
    # where seq[i,j] is the substring of sequence from i to j (inclusive)
    backpointers = {}
    for j in range(len(sequence)): # 
        for i in range(j, -1, -1):
            if "".join(sequence[i:j+1]) in vocabulary:
                opt[i,j] = 1
                backpointers[(i, j)] = [(i, j)]
            else:
                split_costs = {}
                for k in range(i, j):
                    k_split_cost = opt[i,k] + opt[k+1,j]
                    assert k_split_cost < np.inf, f"i: {i}, j: {j}, k: {k}, opt[i,k]: {opt[i,k]}, opt[k+1,j]: {opt[k+1,j]}"
                    split_costs[(i, k, j)] = (k_split_cost)
                opt[i,j] = min(split_costs.values())
                i, k, j = min(split_costs, key=split_costs.get)
                backpointers[(i, j)] = [(i, k), (k+1, j)]

    last_node = (0, len(sequence) - 1)
    best_path = get_best_path(last_node, backpointers)
    return opt[last_node], ["".join(sequence[i:j+1]) for i, j in best_path]


def min_tokens_for_string_quadratic(s, vocabulary):
    n = len(s)
    dp = [float('inf')] * (n + 1)
    dp[0] = 0  # Base case: empty string
    
    for i in range(1, n + 1):
        for j in range(i):
            if s[j:i] in vocabulary:
                dp[i] = min(dp[i], dp[j] + 1)
    return dp[n], ""


def get_model_tokenizer(model_name: str):
    tokenizer_factory = {
        "bert-base-uncased": AutoTokenizer.from_pretrained("bert-base-uncased"),
        "bert-base-multilingual-cased": AutoTokenizer.from_pretrained("bert-base-multilingual-cased"),
        "bigscience/bloom-3b": AutoTokenizer.from_pretrained("bigscience/bloom-3b"),
        "google/mt5-small": AutoTokenizer.from_pretrained("google/mt5-small", legacy=False),
        "google/umt5-small": AutoTokenizer.from_pretrained("google/umt5-small", legacy=False),
        "facebook/xlm-v-base": AutoTokenizer.from_pretrained("facebook/xlm-v-base"),
    }
    if model_name in tokenizer_factory:
        return tokenizer_factory[model_name]
    else:
        msg = f"No model_name {model_name}"
        raise NotImplementedError(msg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--filename", required=True)
    args = parser.parse_args()
    tokenizer = get_model_tokenizer(args.model_name)
    # vocabulary = set(tokenizer.vocab.keys())
    s = OptTokenizer._get_special(args.model_name)
    vocabulary = set(k.lstrip(s) for k, v in tokenizer.vocab.items())
    # if args.filename == "iwslt2017-de-en":
    #     words = read_iwslt(args.filename)
    # elif "LADEC" in args.filename:
    if "LADEC" in args.filename:
        words = read_ladec_data(args.filename, vocabulary)
    else:
        words = read_celex_data(args.filename, vocabulary)
    # TODO: Do we need to append special in front of each char?
    vocabulary = vocabulary.union(*[set(w) for w in words])
    flota_tokenizer = FlotaTokenizer(args.model_name, k=100, strict=False, mode="flota")
    num_chars = []
    base_compression = []
    our_compression = []
    flota_compression = []
    b_f_o_segs = []
    our_tokenizer = OptTokenizer(vocabulary, args.model_name, tokenizer)
    # for word_idx, w in tqdm(enumerate(words)):
    for w in tqdm(words):
        # FIXME: This is specific to BPE uncased
        if "uncased" in args.model_name:
            w = w.lower()
        bpe_tokens = tokenizer.convert_ids_to_tokens(
            tokenizer(w)["input_ids"],
            skip_special_tokens=True,
        )
        # bpe_tokens = [b for b in bpe_tokens if not b == our_tokenizer.special]
        base_compression.append(len(bpe_tokens))
        num_chars.append(len(w))
        vocabulary = vocabulary.union(set(w))
        # num_segs, segmentation = compute_length_of_most_efficient_tokenization(list(w), vocabulary)
        # num_segs, segmentation = min_tokens_for_string_quadratic(w, vocabulary)
        # pretokenized = our_tokenizer.pretokenize(w)[0]
        # segmentation = our_tokenizer.tokenize(pretokenized)
        segmentation = our_tokenizer.tokenize(w)
        # segmentation = [s for s in segmentation if not s == our_tokenizer.special]
        our_compression.append(len(segmentation))
        # num_hyphens = len(re.findall("-", w))
        flota_tokens = flota_tokenizer.tokenize(w)
        # flota_compression.append(len(flota_tokens) + num_hyphens)
        flota_compression.append(len(flota_tokens))
        # print(w)
        # print(bpe_tokens)
        # print(segmentation)
        # print(flota_tokens)
        # print("="*12)
        # if word_idx == 0:
        #     print(bpe_tokens, flota_tokens, segmentation)
        if len(flota_tokens) < len(segmentation):
            print("BUG??", bpe_tokens, flota_tokens, segmentation)
        if len(flota_tokens) > len(segmentation) and len(bpe_tokens) > len(segmentation):
            b_f_o_segs.append((bpe_tokens, flota_tokens, segmentation))

    # for i, (o, f) in enumerate(zip(our_compression, flota_compression)):
    #     if o > f:
    #         print(words[i], b_f_o_segs[i])
    print(args.filename)
    print(args.model_name, np.mean(base_compression), sep=" ")
    print("FLOTA:", np.mean(flota_compression), sep=" ")
    print("OURS:", np.mean(our_compression), sep=" ")
    print("CHAR:", np.mean(num_chars), sep=" ")
    for i, (b, f, o) in enumerate(b_f_o_segs):
        print(words[i], b, f, o)


if __name__ == "__main__":
    main()
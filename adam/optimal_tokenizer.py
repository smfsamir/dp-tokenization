from dataclasses import dataclass
from typing import Dict, List, Set
import re

import torch
import transformers


@dataclass
class OptTokenizer:
    vocab: Set[str]
    model_name: str
    tokenizer: transformers.PreTrainedTokenizer
    max_len: int = None
    disregard_word_initial_marker: bool=False

    def __post_init__(self):
        self.special = self._get_special(self.model_name)
        if not self.max_len:
            self.max_len = self.tokenizer.model_max_length
        if self.disregard_word_initial_marker:
            print("Disregarding word initial space marker")
            self.vocab = {v.lstrip(self.special) for v in self.vocab}

    def obtain_longest_token(self, tokenizations: List[List[str]]) -> List[str]:
        """Obtain the longest tokenization from a list of tokenizations.

        Args:
            tokenizations (List[List[str]]): List of tokenizations to choose from.

        Returns:
            List[str]: The longest tokenization.
        """
        tokenization_lengths = [max([len(t) for t in tokenization]) for tokenization in tokenizations]
        # return the tokenization with the longest length
        return tokenizations[tokenization_lengths.index(max(tokenization_lengths))]


    @staticmethod
    def _get_special(model_name: str):
        special_fac = {
            "bert-base-uncased": "##",
            "bert-base-multilingual-cased": "##",
            "bigscience/bloom-3b": "Ġ",
            "google/mt5-small": "\u2581",
            "google/umt5-small": "\u2581",
        }
        return special_fac[model_name]

    # def get_best_path(self, i: int, backpointers: Dict, s: str, segs = None):
    #     if not segs:
    #         segs = []
    #     if i == 0:
    #         return segs
    #     j, is_special = backpointers[i]
    #     if is_special:
    #         seg = self.special + s[j:i]
    #     else:
    #         seg = s[j:i]
    #     segs.insert(0, seg)
    #     # If i (i.e. index of start of seg) is 0,
    #     # we have the full segmentation
    #     return self.get_best_path(j, backpointers, s, segs)
    
    # def tokenize(self, s: str):
    #     n = len(s)
    #     dp = [float('inf')] * (n + 1)
    #     dp[0] = 0  # Base case: empty string
    #     backpointers = {}
        
    #     for i in range(1, n + 1):
    #         for j in range(i):
    #             # TODO: Apparantly if we default to special v.s. not start with special
    #             #       is model dependent? if s[j:i] in self.vocab:
    #             # TODO: And for most models it seems to be that we first search for the 
    #             #       special start of word char only for the initial chars.
    #             if j == 0:
    #                 if self.special + s[j:i] in self.vocab:
    #                     if dp[j] + 1 < dp[i]:
    #                         dp[i] = dp[j] + 1
    #                         backpointers[i] = (j, True)
    #                 elif s[j:i] in self.vocab:
    #                     if dp[j] + 1 < dp[i]:
    #                         dp[i] = dp[j] + 1
    #                         backpointers[i] = (j, False)
    #             else:
    #                 if s[j:i] in self.vocab:
    #                     if dp[j] + 1 < dp[i]:
    #                         dp[i] = dp[j] + 1
    #                         backpointers[i] = (j, False)
    #                 elif self.special + s[j:i] in self.vocab:
    #                     if dp[j] + 1 < dp[i]:
    #                         dp[i] = dp[j] + 1
    #                         backpointers[i] = (j, True)
    #     # return dp[n]
    #     return self.get_best_path(n, backpointers, s)

    # def pretokenize(self, vocab_bidict):
    #     def pre(input_str):
    #         tokens = self.tokenizer.encode(input_str)
    #         tokens = [vocab_bidict.inverse[token] for token in tokens]
    #         pre_tokens = self.merge_tokens(tokens)
    #         return pre_tokens
    #     return pre
    
    def merge_tokens(self, tokens):
        merged_tokens = []
        i = 0
        while i < len(tokens):
            # Merge tokens that are part of the same word
            if i + 1 < len(tokens) and not tokens[i + 1].startswith(self.special):
                merged_token = tokens[i]
                while i + 1 < len(tokens) and not tokens[i + 1].startswith(self.special):
                    merged_token += tokens[i + 1]
                    i += 1
                merged_tokens.append(merged_token)
            else:
                merged_tokens.append(tokens[i])
            i += 1

        return merged_tokens

    def pretokenize(self, input_str):
        tokens = self.tokenizer.encode(input_str)#re.findall(r'[\w]+|[^\s\w]', input_str)
        tokens = self.tokenizer.convert_ids_to_tokens(tokens, skip_special_tokens=True)
        return self.merge_tokens(tokens)
    
    def tokenize(
        self,
        base_representation_s: List[str],
        max_search_results = 100
     ) -> List[List[str]]:
        """Compute the shortest tokenization of the input string using the provided vocabulary.

        Args:
            base_representation_s (List[str]): The input string to tokenize, represented as a list of atomic tokens
                (e.g., characters for Unigram, bytes for Byte-level BPE, etc.)
            vocabulary (Set[str]): The set of all tokens in the vocabulary.
            disregard_word_initial_marker (bool): Whether to disregard the word initial marker. This means 
                that if we have two tokens that are only different in that one has the word initial marker and the other
                does not, we will treat them as the same token. Set this to true for encoding. But for decoding, 
                you'd want to set this to false in order to be able to decode to a unique, orthographically readable string.
            word_initial_marker (str): The word initial marker. (e.g., '##')
        """
        n = len(base_representation_s)
        len_dp = [float('inf')] * (n + 1)
        len_dp[0] = 0  # Base case: empty string
        segment_index_dp = [[i] for i in range(n)]
        
        def _get_concatenation(slice: List[str]):
            return "".join(slice)

        for i in range(1, n + 1):
            min_split_indices = []
            prev_min_length = len_dp[i]
            for j in range(i):
                # if j == 0 and _get_concatenation(
                #     self.special + base_representation_s[j:i]
                # ) in self.vocab:
                #     if len_dp[j] + 1 < len_dp[i]:
                #         if len_dp[i] > len_dp[j] + 1:
                #             len_dp[i] = len_dp[j] + 1
                #             if len_dp[i] < prev_min_length:
                #                 min_split_indices = [j]
                #         if len_dp[i] == len_dp[j] + 1:
                #             if j not in min_split_indices: # we should be able to do without this, idk
                #                 min_split_indices.append(j)
                if _get_concatenation(base_representation_s[j:i]) in self.vocab:
                    if len_dp[i] > len_dp[j] + 1:
                        len_dp[i] = len_dp[j] + 1
                        if len_dp[i] < prev_min_length:
                            min_split_indices = [j]
                    if len_dp[i] == len_dp[j] + 1:
                        if j not in min_split_indices: # we should be able to do without this, idk
                            min_split_indices.append(j)
            segment_index_dp[i-1] = min_split_indices

        
        backtrace_indices = segment_index_dp[-1].copy()
        curr_indices = [] 
        for i in range(len(segment_index_dp[-1])):
            curr_indices.append(n)
        tokenizations = [] 
        for i in range(len(segment_index_dp[-1])):
            tokenizations.append([])
        complete_tokenizations = []
        while backtrace_indices:
            backtrace_index = backtrace_indices.pop()
            curr_index = curr_indices.pop() - 1
            curr_tokenization = tokenizations.pop()
            # if not curr_index in discovered:
            curr_tokenization.insert(0, ''.join(base_representation_s[backtrace_index:curr_index + 1]))
            if 0 in segment_index_dp[curr_index]:
                complete_tokenizations.append(curr_tokenization)
                if len(complete_tokenizations) > max_search_results:
                    break
            else:
                backtrace_indices.extend(segment_index_dp[backtrace_index - 1].copy())
                curr_indices.extend(len(segment_index_dp[backtrace_index - 1]) * [backtrace_index])
                for i in range(len(segment_index_dp[backtrace_index - 1])):
                    tokenizations.append(curr_tokenization.copy()) 
        # return complete_tokenizations, len_dp[-1]
        # TODO: By default we just return the one with the largest substring.
        return self.obtain_longest_token(complete_tokenizations)

    # Taken from flota
    def encode(self, text: List[str]):
        """Taken from flota."""
        # Pretokenizes: splits on whitespaces and non-alphanumerics.
        # text_split = re.findall(r'[\w]+|[^\s\w]', text)
        text_split = self.pretokenize(text)
        tokens = list()
        for w in text_split:
            tokens.extend(self.tokenize(w))
        if self.model_name in ["bert-base-cased", "bert-base-uncased", "bert-base-multilingual-cased"]:
            ids = self.tokenizer.convert_tokens_to_ids(tokens)[:self.max_len - 2]
            return [self.tokenizer.cls_token_id] + ids + [self.tokenizer.sep_token_id]
        elif self.model_name in ['bigscience/bloom-3b', 'gpt2', 'google/umt5-small', 'google/mt5-small']:
            ids = self.tokenizer.convert_tokens_to_ids(tokens) + [self.tokenizer.eos_token_id]
            return ids[:self.max_len]
        elif self.model_name in ['xlnet-base-cased']:
            ids = self.tokenizer.convert_tokens_to_ids(tokens)[:self.max_len - 2]
            return ids + [self.tokenizer.sep_token_id, self.tokenizer.cls_token_id]

    # Taken from flota
    def __call__(self, texts):
        """Taken from flota"""
        if isinstance(texts, str):
            texts = [texts]
        texts = [self.encode(text) for text in texts]
        batch_size = len(texts)
        max_len = max(len(text) for text in texts)
        if self.model_name  in ["bert-base-cased", "bert-base-uncased", "bert-base-multilingual-cased"]:
            input_ids = torch.zeros((batch_size, max_len)).long()
            attention_mask = torch.zeros((batch_size, max_len)).long()
            for i, text in enumerate(texts):
                input_ids[i, :len(text)] = torch.tensor(text)
                attention_mask[i, :len(text)] = 1
        elif self.model_name in ['bigscience/bloom-3b', 'gpt2']:
            input_ids = self.tokenizer.eos_token_id * torch.ones((batch_size, max_len)).long()
            attention_mask = torch.zeros((batch_size, max_len)).long()
            for i, text in enumerate(texts):
                input_ids[i, -len(text):] = torch.tensor(text)
                attention_mask[i, -len(text):] = 1
        elif self.model_name in ['google/umt5-small', 'google/mt5-small']:
            input_ids = self.tokenizer.pad_token_id * torch.ones((batch_size, max_len)).long()
            attention_mask = torch.zeros((batch_size, max_len)).long()
            for i, text in enumerate(texts):
                input_ids[i, :len(text)] = torch.tensor(text)
                attention_mask[i, :len(text)] = 1
        elif self.model_name in ['xlnet-base-cased']:
            input_ids = self.tokenizer.pad_token_id * torch.ones((batch_size, max_len)).long()
            attention_mask = torch.zeros((batch_size, max_len)).long()
            for i, text in enumerate(texts):
                input_ids[i, -len(text):] = torch.tensor(text)
                attention_mask[i, -len(text):] = 1
        
        if batch_size == 1:
            input_ids = input_ids[0]
            attention_mask = attention_mask[0]
        tensors = {'input_ids': input_ids, 'attention_mask': attention_mask}
        return tensors

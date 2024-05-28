import pdb
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List
import math


def activation(act):
    if act == 'relu':
        return nn.ReLU()
    elif act == 'silu':
        return nn.SiLU()
    else:
        return nn.GELU()


def exact_match(predictions, scores, targets, total_sequence_generated):
    batched_predictions = []
    corr_list = []
    batch_length = len(targets)
    for b in range(batch_length):
        one_batch_sequence = predictions[b * total_sequence_generated: (b + 1) * total_sequence_generated]
        one_batch_score = scores[b * total_sequence_generated: (b + 1) * total_sequence_generated]
        pairs = [(a, b) for a, b in zip(one_batch_sequence, one_batch_score)]
        sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
        batched_predictions.append(one_batch_sequence)
    hit_1 = 0
    hit_5 = 0
    hit_10 = 0
    ndcg_5 = 0
    ndcg_10 = 0
    for pred, target in zip(batched_predictions, targets):
        pred_10 = [one_p for one_p in pred[:10]]
        # hit@k
        if target in pred_10[:1]:
            hit_1 += 1
        if target in pred_10[:5]:
            hit_5 += 1
        if target in pred_10:
            hit_10 += 1
            corr_list.append(1)
        else:
            corr_list.append(0)
        if target in pred_10[:5]:
            gold_position = pred_10[:5].index(target)
            true_scores = [0.0] * 5
            true_scores[gold_position] = 1.0
            ndcg_5 += sum([score / math.log2(1 + idx) for idx, score in enumerate(true_scores, start=1)]) / 1
        else:
            ndcg_5 += 0
        if target in pred_10:
            gold_position = pred_10.index(target)
            true_scores = [0.0] * 10
            true_scores[gold_position] = 1.0
            ndcg_10 += sum([score / math.log2(1 + idx) for idx, score in enumerate(true_scores, start=1)]) / 1
        else:
            ndcg_10 += 0

    return hit_1, hit_5, hit_10, ndcg_5, ndcg_10, corr_list


def prefix_allowed_tokens_fn(candidate_trie):
    def prefix_allowed_tokens(batch_id, sentence):
        sentence = sentence.tolist()
        trie_out = candidate_trie.get(sentence)
        return trie_out

    return prefix_allowed_tokens


def predict_outputs(batch, model, prefix_allowed_tokens=None, k=20, max_len=20, tokenizer=None):
    device = next(model.parameters()).device
    lm_labels = batch["target_ids"].to(device)
    prediction = model.generate_step(batch, k=k, max_len=max_len, constraint=prefix_allowed_tokens)
    lm_labels = torch.where(lm_labels == -100, 0, lm_labels)
    gold_sents = tokenizer.batch_decode(lm_labels, skip_special_tokens=True)
    generated_sents = tokenizer.batch_decode(prediction['sequences'], skip_special_tokens=True)
    hit_1, hit_5, hit_10, ncdg_5, ncdg_10, corr_list = exact_match(generated_sents, prediction['sequences_scores'], gold_sents, k)
    return hit_1, hit_5, hit_10, ncdg_5, ncdg_10, corr_list


def save_outputs(batch, model, prefix_allowed_tokens=None, k=20, max_len=20, tokenizer=None):
    pred_outs = []
    device = next(model.parameters()).device
    lm_labels = batch["target_ids"].to(device)
    label_length = batch['target_length'].to(device)
    prediction = model.generate_step(batch, k=k, max_len=max_len, constraint=prefix_allowed_tokens)
    lm_labels = torch.where(lm_labels == -100, 0, lm_labels)
    gold_sents = tokenizer.batch_decode(lm_labels, skip_special_tokens=True)
    generated_sents = tokenizer.batch_decode(prediction['sequences'], skip_special_tokens=True)
    batch_length = len(gold_sents)
    for b in range(batch_length):
        one_batch_sequence = generated_sents[b * k: (b + 1) * k]
        pred_outs.append([gold_sents[b], one_batch_sequence])
    return pred_outs


class Trie(object):
    def __init__(self, sequences: List[List[int]] = []):
        self.trie_dict = {}
        self.len = 0
        if sequences:
            for sequence in sequences:
                Trie._add_to_trie(sequence, self.trie_dict)
                self.len += 1

        self.append_trie = None
        self.bos_token_id = None

    def append(self, trie, bos_token_id):
        self.append_trie = trie
        self.bos_token_id = bos_token_id

    def add(self, sequence: List[int]):
        Trie._add_to_trie(sequence, self.trie_dict)
        self.len += 1

    def get(self, prefix_sequence: List[int]):
        return Trie._get_from_trie(
            prefix_sequence, self.trie_dict, self.append_trie, self.bos_token_id
        )

    @staticmethod
    def load_from_dict(trie_dict):
        trie = Trie()
        trie.trie_dict = trie_dict
        trie.len = sum(1 for _ in trie)
        return trie

    @staticmethod
    def _add_to_trie(sequence: List[int], trie_dict: Dict):
        if sequence:
            if sequence[0] not in trie_dict:
                trie_dict[sequence[0]] = {}
            Trie._add_to_trie(sequence[1:], trie_dict[sequence[0]])

    @staticmethod
    def _get_from_trie(
            prefix_sequence: List[int],
            trie_dict: Dict,
            append_trie=None,
            bos_token_id: int = None,
    ):
        if len(prefix_sequence) == 0:
            output = list(trie_dict.keys())
            if append_trie and bos_token_id in output:
                output.remove(bos_token_id)
                output += list(append_trie.trie_dict.keys())
            return output
        elif prefix_sequence[0] in trie_dict:
            return Trie._get_from_trie(
                prefix_sequence[1:],
                trie_dict[prefix_sequence[0]],
                append_trie,
                bos_token_id,
            )
        else:
            if append_trie:
                return append_trie.get(prefix_sequence)
            else:
                return []

    def __iter__(self):
        def _traverse(prefix_sequence, trie_dict):
            if trie_dict:
                for next_token in trie_dict:
                    yield from _traverse(
                        prefix_sequence + [next_token], trie_dict[next_token]
                    )
            else:
                yield prefix_sequence

        return _traverse([], self.trie_dict)

    def __len__(self):
        return self.len

    def __getitem__(self, value):
        return self.get(value)

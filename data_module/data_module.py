from functools import partial
import os
import json
from dataclasses import dataclass
from random import random
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from collections import defaultdict
from enum import Enum
import time
import random
from sklearn import neighbors
import torch
from sklearn.cluster import KMeans

from torch.utils.data import DataLoader, TensorDataset, Dataset
from transformers import AutoTokenizer, BertTokenizer, T5Tokenizer, T5TokenizerFast
import numpy as np
from .processor import KGCDataset, PretrainKGCDataset
from .base_data_module import BaseKGQADataModule, Config
from .utils import LinkGraph, Roberta_utils
from datasets import Dataset as HFDataset 

ENTITY_PADDING_INDEX = 1

def lmap(f, x):
    return list(map(f, x))



from datasets import Dataset as HFDataset


def process_triplet_batch(examples, tokenizer, relation2text, entity2text, filter_hr_to_t, filter_tr_to_h, st_entity, st_relation, args):
    """Processes a batch of triplet examples."""

    results = {  # Initialize results dictionary
        "kge_input_ids": [],
        "question": [],
        "label": [],
        "label_ids": [],
    }

    for i in range(len(examples['h'])): # Iterate through each example in the batch
        h = examples['h'][i]
        r = examples['r'][i]
        t = examples['t'][i]
        inverse = examples['inverse'][i]
        head_entity = tail_entity = entity2text[h].split(" , ")[0]
        
        if not inverse:
            question = f"What is the {relation2text[r]} of {head_entity}?"
            filter_entities = filter_hr_to_t[(h, r)]
            input_text = [tokenizer.pad_token, entity2text[h], tokenizer.pad_token, relation2text[r], tokenizer.mask_token]
            ent_pos, rel_pos = 1, 3
        else:
            question = f"What is {relation2text[r]} to {tail_entity}?"
            filter_entities = filter_tr_to_h[(h, r)]
            input_text = [tokenizer.mask_token, tokenizer.pad_token, relation2text[r], tokenizer.pad_token, entity2text[h]]
            ent_pos, rel_pos = 4, 2

        kge_input = tokenizer(
            tokenizer.sep_token.join(input_text),
            padding='max_length',  # Pad to max_length
            truncation=True,
            max_length=args.max_seq_length,
            return_tensors="pt"
        )

        kge_input.input_ids[0][ent_pos] = st_entity + (h if not inverse else t)
        kge_input.input_ids[0][rel_pos] = st_relation + r

        # filtered_entities = [ent for ent in filter_entities if ent != t]
        target_entities = [entity2text[ent_id].split(" , ")[0] for ent_id in filter_entities]
        label_text = f"The possible answers: {', '.join(target_entities)}"

        # Append results to the lists
        results["kge_input_ids"].append(kge_input.input_ids[0])
        results["question"].append(question)
        results["label"].append(label_text)
        results["label_ids"].append(filter_entities)
    # Convert lists to tensors (important for batching)
    # results["kge_input_ids"] = torch.stack(results["kge_input_ids"])
    return results


class FB15k237DataModule(BaseKGQADataModule):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.kge_tokenizer = AutoTokenizer.from_pretrained(self.args.model_name_or_path, use_fast=True)
        self.ntokenizer = AutoTokenizer.from_pretrained(self.args.text_encoder_model_name_or_path, use_fast=True)
        self.filter_entity_ids_list = []
        entity_list = [f"[entity{i}]" for i in range(self.num_entity)]
        relation_list = [f"[relation{i}]" for i in range(self.num_relation)]
        self.st_entity = self.kge_tokenizer.vocab_size
        self.ed_entity = self.kge_tokenizer.vocab_size + self.num_entity
        self.st_relation = self.kge_tokenizer.vocab_size + self.num_entity
        self.ed_relation = self.kge_tokenizer.vocab_size + self.num_entity + self.num_relation

    @staticmethod
    def add_to_argparse(parser):
        BaseKGQADataModule.add_to_argparse(parser)
        parser.add_argument("--model_name_or_path", type=str, default="roberta-base", help="the name or the path to the pretrained model")
        parser.add_argument("--text_encoder_model_name_or_path", type=str, default="roberta-base", help="the name or the path to the text encoder pretrained model")
        parser.add_argument("--max_seq_length", type=int, default=256, help="Number of examples to operate on per forward step.")
        parser.add_argument("--eval_batch_size", type=int, default=8)
        parser.add_argument("--overwrite_cache", action="store_true", default=False)
        parser.add_argument("--max_entity_length", type=int, default=256)
        
        return parser
    


    def _create_hf_dataset(self, data_list):
        """Creates a Hugging Face Dataset from a list of dictionaries."""
        # return HFDataset.from_dict({
        #     "kge_input_ids": [data["kge_input_ids"] for data in data_list],
        #     "question": [data["question"] for data in data_list],
        #     "label": [data["label"] for data in data_list],
        # })
        return HFDataset.from_dict(datalist)

    def setup(self, stage=None):
        super().setup(stage)
        tokenizer = AutoTokenizer.from_pretrained(self.args.model_name_or_path, use_fast=True)

        process_func = partial(process_triplet_batch, tokenizer=tokenizer, relation2text=self.relation2text,
                              entity2text=self.entity2text, filter_hr_to_t=self.filter_hr_to_t,
                              filter_tr_to_h=self.filter_tr_to_h, st_entity=self.st_entity,
                              st_relation=self.st_relation, args=self.args)

        for split in ["train", "val", "test"]:
            dataset_dict = {
                "h": [item.hr[0] for item in getattr(self, f"data_{split}")],
                "r": [item.hr[1] for item in getattr(self, f"data_{split}")],
                "t": [item.t for item in getattr(self, f"data_{split}")],
                "inverse": [item.inverse for item in getattr(self, f"data_{split}")]
            }
            dataset = HFDataset.from_dict(dataset_dict).map(
                process_func,
                batched=True,
                batch_size=2048,
                # num_proc=self.args.preprocessing_num_workers, 
                with_indices=False,
                desc=f"Processing {split} dataset"
            )
            dataset.to_json(f"./Processed/FB15k-237_roberta/{split}_dataset.jsonl")
            setattr(self, f"{split}_dataset", dataset)

    def _process_data_row(self, example):
        """Processes a single row of the dataset."""
        h = example['h']
        r = example['r']
        t = example['t']
        inverse = example['inverse']

        return self._process_triplet(h, r, t, inverse)
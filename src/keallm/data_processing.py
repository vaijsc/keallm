import json
import random
from typing import List, Tuple, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

class KnowledgeGraphDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: BertTokenizer):
        self.tokenizer = tokenizer
        self.triples = self.load_triples(data_path)

    def load_triples(self, data_path: str) -> List[Tuple[str, str, str]]:
        with open(data_path, "r") as f:
            triples = [tuple(line.strip().split("\t")) for line in f]
        return triples

    def __len__(self) -> int:
        return len(self.triples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        head, relation, tail = self.triples[idx]
        input_ids = self.tokenizer.encode(
            f"[CLS] {head} [SEP] {relation} [SEP] [MASK] [SEP]", add_special_tokens=True
        )
        attention_mask = [1] * len(input_ids)
        label = self.tokenizer.encode(tail, add_special_tokens=False)[0]
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "label": torch.tensor(label),
        }

class ProjectorDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        kg_embedding_model,
        tokenizer: BertTokenizer,
        template: str = "What is the [relation] of [head entity]?",
    ):
        self.kg_embedding_model = kg_embedding_model
        self.tokenizer = tokenizer
        self.template = template
        self.data = self.load_data(data_path)

    def load_data(self, data_path: str) -> List[Dict[str, torch.Tensor]]:
        with open(data_path, "r") as f:
            data = [json.loads(line) for line in f]
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        head = item["head"]
        relation = item["relation"]
        tail = item["tail"]
        query = self.template.replace("[relation]", relation).replace(
            "[head entity]", head
        )
        query_ids = self.tokenizer.encode(query, add_special_tokens=True)
        kg_embedding, non_padding_positions = self.kg_embedding_model.get_embedding(head, relation)
        label_id = self.tokenizer.encode(tail, add_special_tokens=False)[0]
        return {
            "query_ids": torch.tensor(query_ids),
            "kg_embedding": torch.tensor(kg_embedding),
            "non_padding_positions": torch.tensor(non_padding_positions),
            "label_id": torch.tensor(label_id),
        }

def create_projector_dataset(
    kg_path: str,
    kg_embedding_model,
    tokenizer: BertTokenizer,
    output_path: str,
    num_samples: int = 15000,
    template: str = "What is the [relation] of [head entity]?",
):
    with open(kg_path, "r") as f:
        triples = [tuple(line.strip().split("\t")) for line in f]
    random.shuffle(triples)
    triples = triples[:num_samples]
    data = []
    for head, relation, tail in triples:
        query = template.replace("[relation]", relation).replace("[head entity]", head)
        data.append({"head": head, "relation": relation, "tail": tail, "query": query})
    with open(output_path, "w") as f:
        for item in data:
            json.dump(item, f)
            f.write("\n")
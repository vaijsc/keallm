import json
import random
from typing import List, Tuple, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, AutoTokenizer

class KnowledgeGraphDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: BertTokenizer):
        self.tokenizer = tokenizer
        self.triples = self.load_triples(data_path)

    def load_triples(self, data_path: str) -> List[Tuple[str, str, str]]:
        with open(data_path, "r") as f:
            triples = [tuple(line.strip().split("|")) for line in f]
        return triples

    def __len__(self) -> int:
        return len(self.triples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        head, relation, tail = self.triples[idx]
        input_ids = self.tokenizer.encode(
            f"[CLS] {head} [SEP] {relation} [SEP] [MASK] [SEP]", add_special_tokens=True, truncation=True, max_length=512
        )
        attention_mask = [1] * len(input_ids)
        label = self.tokenizer.encode(tail, add_special_tokens=False)[0]
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(label),
        }

class MetaQAProjectorDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        kg_embedding_model,
        tokenizer: AutoTokenizer,
        hops: int = 1,
    ):
        assert hops in [1, 2, 3], "Hops should be 1, 2, or 3"
        self.kg_embedding_model = kg_embedding_model
        self.tokenizer = tokenizer
        self.hops = hops
        with open(data_path, "r") as f:
            self.data = json.load(f)

    def __len__(self) -> int:
        return len(self.data["Questions"])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        question = self.data["Questions"][idx]["Question"]
        answers = [a["AnswerArgument"] for a in self.data["Questions"][idx]["Answers"]]
        head = self.data["Questions"][idx]["Parse"][0]["Subject"]
        relation = self.data["Questions"][idx]["Parse"][0]["Relation"]
        if self.hops > 1:
            for i in range(1, self.hops):
                relation += " " + self.data["Questions"][idx]["Parse"][i]["Relation"]

        kg_embedding, non_padding_positions = self.kg_embedding_model.get_embedding(head, relation)
        
        # Tokenize the question and answers
        encoded_question = self.tokenizer(
            question,
            add_special_tokens=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        
        # Find the index of the correct answer in the list of answers
        correct_answer_index = answers.index(self.data["Questions"][idx]["Answers"][0]["AnswerArgument"])
        
        return {
            "input_ids": encoded_question["input_ids"].squeeze(0),
            "attention_mask": encoded_question["attention_mask"].squeeze(0),
            "kg_embedding": torch.tensor(kg_embedding),
            "non_padding_positions": torch.tensor(non_padding_positions),
            "labels": torch.tensor(correct_answer_index),
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

def add_kg_vocab_to_tokenizer(tokenizer: AutoTokenizer, kg_path: str):
    """Adds entities and relations from the KG to the tokenizer's vocabulary."""
    with open(kg_path, "r") as f:
        for line in f:
            head, relation, tail = line.strip().split("|")
            tokenizer.add_tokens([head, relation, tail])
    return tokenizer
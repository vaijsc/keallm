import argparse
import json
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, LlamaForCausalLM

from .data_processing import ProjectorDataset
from .model import KEALLM, KGembeddingModel

def evaluate(
    model: KEALLM,
    kg_embedding_model: KGembeddingModel,
    tokenizer: BertTokenizer,
    data_path: str,
    batch_size: int = 16,
) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = ProjectorDataset(data_path, kg_embedding_model, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            query_ids = batch["query_ids"].to(device)
            kg_embedding = batch["kg_embedding"].to(device)
            non_padding_positions = batch["non_padding_positions"].to(device)
            labels = batch["label_id"].to(device)
            outputs = model(query_ids, kg_embedding, non_padding_positions)
            predicted_labels = torch.argmax(outputs["logits"], dim=-1)[:, -1]
            correct += (predicted_labels == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    return accuracy
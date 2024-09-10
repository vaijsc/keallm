import argparse
from typing import Dict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from .model import KEALLM, KGembeddingModel

def answer_question(
    model: KEALLM,
    kg_embedding_model: KGembeddingModel,
    tokenizer: AutoTokenizer,
    query: str,
    head: str,
    relation: str,
    max_new_tokens: int = 100,
) -> str:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    query_ids = tokenizer.encode(query, return_tensors="pt", add_special_tokens=True).to(device)
    kg_embedding, non_padding_positions = kg_embedding_model.get_embedding(head, relation)
    kg_embedding = kg_embedding.to(device)
    non_padding_positions = non_padding_positions.to(device)
    generated_ids = model.llama_model.generate(
        input_ids=query_ids,
        kg_embedding=kg_embedding,
        non_padding_positions=non_padding_positions,
        max_new_tokens=max_new_tokens,
    )
    answer = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return answer
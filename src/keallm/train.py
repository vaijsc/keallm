import argparse
import logging
from typing import Dict

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    BertModel,
    BertTokenizer,
    LlamaForCausalLM,
    get_linear_schedule_with_warmup,
)

from .data_processing import KnowledgeGraphDataset, ProjectorDataset
from .model import KEALLM, KGembeddingModel

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def train_kg_embedding(
    kg_path: str,
    model: KGembeddingModel,
    tokenizer: BertTokenizer,
    learning_rate: float = 2e-5,
    batch_size: int = 16,
    epochs: int = 5,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    dataset = KnowledgeGraphDataset(kg_path, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    total_steps = len(dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            model.zero_grad()
            outputs = model(input_ids, attention_mask, labels=labels)
            loss = outputs["loss"]
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
            if step % 100 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs}, Step {step+1}/{len(dataloader)}, Loss: {loss.item():.4f}"
                )
        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
    return model

def train_projector(
    projector_data_path: str,
    model: KEALLM,
    kg_embedding_model: KGembeddingModel,
    tokenizer: BertTokenizer,
    learning_rate: float = 2e-5,
    batch_size: int = 16,
    epochs: int = 10,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = AdamW(model.projector.parameters(), lr=learning_rate)
    dataset = ProjectorDataset(projector_data_path, kg_embedding_model, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    total_steps = len(dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(dataloader):
            query_ids = batch["query_ids"].to(device)
            kg_embedding = batch["kg_embedding"].to(device)
            non_padding_positions = batch["non_padding_positions"].to(device)
            labels = batch["label_id"].to(device)
            model.zero_grad()
            outputs = model(query_ids, kg_embedding, non_padding_positions, labels=labels)
            loss = outputs["loss"]
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
            if step % 100 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs}, Step {step+1}/{len(dataloader)}, Loss: {loss.item():.4f}"
                )
        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
    return model

def train_dpo(
    projector_data_path: str,
    model: KEALLM,
    kg_embedding_model: KGembeddingModel,
    tokenizer: BertTokenizer,
    reference_model: KEALLM,
    learning_rate: float = 5e-6,
    batch_size: int = 8,
    epochs: int = 5,
    beta: float = 0.1,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    reference_model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    dataset = ProjectorDataset(projector_data_path, kg_embedding_model, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(dataloader):
            query_ids = batch["query_ids"].to(device)
            kg_embedding = batch["kg_embedding"].to(device)
            non_padding_positions = batch["non_padding_positions"].to(device)
            preferred_label_id = batch["label_id"].to(device)
            # Get less preferred entity from KG embedding model
            less_preferred_entity = kg_embedding_model.get_least_likely_entity(
                batch["head"][0], batch["relation"][0]
            )
            less_preferred_label_id = tokenizer.encode(
                less_preferred_entity, add_special_tokens=False
            )[0].to(device)
            model.zero_grad()
            preferred_outputs = model(query_ids, kg_embedding, non_padding_positions, labels=preferred_label_id)
            less_preferred_outputs = model(
                query_ids, kg_embedding, non_padding_positions, labels=less_preferred_label_id
            )
            preferred_log_probs = torch.log_softmax(preferred_outputs["logits"], dim=-1)
            less_preferred_log_probs = torch.log_softmax(
                less_preferred_outputs["logits"], dim=-1
            )
            preferred_prob = preferred_log_probs[:, -1, preferred_label_id].mean()
            less_preferred_prob = less_preferred_log_probs[:, -1, less_preferred_label_id].mean()
            with torch.no_grad():
                reference_preferred_prob = torch.log_softmax(
                    reference_model(query_ids, kg_embedding, non_padding_positions)["logits"], dim=-1
                )[:, -1, preferred_label_id].mean()
                reference_less_preferred_prob = torch.log_softmax(
                    reference_model(query_ids, kg_embedding, non_padding_positions)["logits"], dim=-1
                )[:, -1, less_preferred_label_id].mean()
            dpo_loss = -(
                torch.sigmoid(
                    beta
                    * (
                        preferred_prob
                        - less_preferred_prob
                        - reference_preferred_prob
                        + reference_less_preferred_prob
                    )
                )
            ).mean()
            total_loss += dpo_loss.item()
            dpo_loss.backward()
            optimizer.step()
            if step % 100 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs}, Step {step+1}/{len(dataloader)}, Loss: {dpo_loss.item():.4f}"
                )
        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
    return model
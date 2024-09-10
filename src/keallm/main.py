import argparse
import logging
import os
from transformers import BertModel, BertTokenizer, AutoModelForCausalLM

from .data_processing import create_projector_dataset
from .model import KEALLM, KGembeddingModel
from .train import train_kg_embedding, train_projector, train_dpo
from .evaluate import evaluate
from .inference import answer_question

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(
        description="Knowledge Embedding Augmented Large Language Model"
    )
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-2-7b-chat-hf", help="Path to LLM model")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train KG Embedding command
    train_kg_parser = subparsers.add_parser("train_kg", help="Train KG embedding model")
    train_kg_parser.add_argument(
        "--kg_path", type=str, required=True, help="Path to knowledge graph data"
    )
    train_kg_parser.add_argument(
        "--output_path", type=str, required=True, help="Path to save trained model"
    )

    # Train Projector command
    train_projector_parser = subparsers.add_parser(
        "train_projector", help="Train Projector module"
    )
    train_projector_parser.add_argument(
        "--projector_data_path",
        type=str,
        required=True,
        help="Path to Projector training data",
    )
    train_projector_parser.add_argument(
        "--kg_embedding_path",
        type=str,
        required=True,
        help="Path to trained KG embedding model",
    )
    train_projector_parser.add_argument(
        "--output_path", type=str, required=True, help="Path to save trained model"
    )

    # Train DPO command
    train_dpo_parser = subparsers.add_parser(
        "train_dpo", help="Train KEALLM with DPO"
    )
    train_dpo_parser.add_argument(
        "--projector_data_path",
        type=str,
        required=True,
        help="Path to Projector training data",
    )
    train_dpo_parser.add_argument(
        "--kg_embedding_path",
        type=str,
        required=True,
        help="Path to trained KG embedding model",
    )
    train_dpo_parser.add_argument(
        "--reference_model_path",
        type=str,
        required=True,
        help="Path to reference KEALLM model",
    )
    train_dpo_parser.add_argument(
        "--output_path", type=str, required=True, help="Path to save trained model"
    )

    # Evaluate command
    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Evaluate KEALLM on question answering"
    )
    evaluate_parser.add_argument(
        "--model_path", type=str, required=True, help="Path to trained KEALLM model"
    )
    evaluate_parser.add_argument(
        "--kg_embedding_path",
        type=str,
        required=True,
        help="Path to trained KG embedding model",
    )
    evaluate_parser.add_argument(
        "--data_path", type=str, required=True, help="Path to evaluation data"
    )

    # Inference command
    inference_parser = subparsers.add_parser("inference", help="Answer a question")
    inference_parser.add_argument(
        "--model_path", type=str, required=False, default="", help="Path to trained KEALLM model"
    )
    inference_parser.add_argument(
        "--kg_embedding_path",
        type=str,
        required=False,
        default="",
        help="Path to trained KG embedding model",
    )
    inference_parser.add_argument(
        "--query", type=str, required=True, help="User query"
    )
    inference_parser.add_argument(
        "--head", type=str, required=True, help="Head entity in the knowledge graph"
    )
    inference_parser.add_argument(
        "--relation", type=str, required=True, help="Relation in the knowledge graph"
    )

    # Create Projector Dataset command
    create_projector_dataset_parser = subparsers.add_parser(
        "create_projector_dataset", help="Create training dataset for Projector module"
    )
    create_projector_dataset_parser.add_argument(
        "--kg_path", type=str, required=True, help="Path to knowledge graph data"
    )
    create_projector_dataset_parser.add_argument(
        "--kg_embedding_path",
        type=str,
        required=True,
        help="Path to trained KG embedding model",
    )
    create_projector_dataset_parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the created dataset",
    )

    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    llama_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)

    if args.command == "train_kg":
        bert_model = BertModel.from_pretrained("bert-base-uncased")
        kg_embedding_model = KGembeddingModel(bert_model, tokenizer)
        trained_kg_embedding_model = train_kg_embedding(
            args.kg_path, kg_embedding_model, tokenizer
        )
        torch.save(trained_kg_embedding_model.state_dict(), args.output_path)
        logger.info(f"Trained KG embedding model saved to {args.output_path}")

    elif args.command == "train_projector":
        kg_embedding_model = KGembeddingModel(BertModel.from_pretrained("bert-base-uncased"), tokenizer)
        kg_embedding_model.load_state_dict(torch.load(args.kg_embedding_path))
        model = KEALLM(llama_model, kg_embedding_model.bert_model.config.hidden_size, llama_model.config.hidden_size)
        trained_model = train_projector(
            args.projector_data_path, model, kg_embedding_model, tokenizer
        )
        torch.save(trained_model.state_dict(), args.output_path)
        logger.info(f"Trained Projector module saved to {args.output_path}")

    elif args.command == "train_dpo":
        kg_embedding_model = KGembeddingModel(BertModel.from_pretrained("bert-base-uncased"), tokenizer)
        kg_embedding_model.load_state_dict(torch.load(args.kg_embedding_path))
        model = KEALLM(llama_model, kg_embedding_model.bert_model.config.hidden_size, llama_model.config.hidden_size)
        model.load_state_dict(torch.load(args.model_path))
        reference_model = KEALLM(llama_model, kg_embedding_model.bert_model.config.hidden_size, llama_model.config.hidden_size)
        reference_model.load_state_dict(torch.load(args.reference_model_path))
        trained_model = train_dpo(
            args.projector_data_path,
            model,
            kg_embedding_model,
            tokenizer,
            reference_model,
        )
        torch.save(trained_model.state_dict(), args.output_path)
        logger.info(f"Trained KEALLM model with DPO saved to {args.output_path}")

    elif args.command == "evaluate":
        kg_embedding_model = KGembeddingModel(BertModel.from_pretrained("bert-base-uncased"), tokenizer)
        kg_embedding_model.load_state_dict(torch.load(args.kg_embedding_path))
        model = KEALLM(llama_model, kg_embedding_model.bert_model.config.hidden_size, llama_model.config.hidden_size)
        model.load_state_dict(torch.load(args.model_path))
        accuracy = evaluate(model, kg_embedding_model, tokenizer, args.data_path)
        logger.info(f"Accuracy: {accuracy:.4f}")

    elif args.command == "inference":
        kg_embedding_model = KGembeddingModel(BertModel.from_pretrained("bert-base-uncased"), tokenizer)
        if os.path.exists(args.kg_embedding_path):
            kg_embedding_model.load_state_dict(torch.load(args.kg_embedding_path))
        model = KEALLM(llama_model, kg_embedding_model.bert_model.config.hidden_size, llama_model.config.hidden_size)
        if os.path.exists(args.model_path):
            model.load_state_dict(torch.load(args.model_path))
        answer = answer_question(
            model,
            kg_embedding_model,
            tokenizer,
            args.query,
            args.head,
            args.relation,
        )
        logger.info(f"Answer: {answer}")

    elif args.command == "create_projector_dataset":
        kg_embedding_model = KGembeddingModel(BertModel.from_pretrained("bert-base-uncased"), tokenizer)
        kg_embedding_model.load_state_dict(torch.load(args.kg_embedding_path))
        create_projector_dataset(
            args.kg_path, kg_embedding_model, tokenizer, args.output_path
        )
        logger.info(f"Projector dataset created at {args.output_path}")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
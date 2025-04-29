import argparse
import torch
from transformers import AutoModel,AutoTokenizer,AutoConfig
from torch.utils.data import DataLoader
from torch.optim import AdamW
import wandb
import os
from exercise_representation.math_dataset import MathDataset
from exercise_representation.collate_functions import custom_collate_fn
from exercise_representation.trainer import Trainer
from exercise_representation.evaluator import Evaluator

def main(args):
    wandb.init(project=args.wandb_project_name, config=args)

    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    model_path = ''
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.add_tokens(['[Q]', '[S]', '[KC]'])

    config = AutoConfig.from_pretrained(model_path, hidden_dropout_prob=args.dropout,
                                        attention_probs_dropout_prob=args.dropout)
    model = AutoModel.from_pretrained(model_path, config=config)
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)
    dataset = MathDataset(tokenizer, args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    evaluator = Evaluator(tokenizer, device, args)
    trainer = Trainer(model, dataloader, optimizer, evaluator, device, args)

    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    tokenizer_save_path = os.path.join(args.model_save_dir, 'tokenizer')
    tokenizer.save_pretrained(tokenizer_save_path)
    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a model on Math Dataset")
    parser.add_argument('--json_file_dataset', type=str, required=True,
                        help='Path to the JSON file containing the dataset')
    parser.add_argument('--json_file_cluster_kc', type=str, required=True,
                        help='Path to the JSON file containing KC cluster mappings')
    parser.add_argument('--json_file_kc_questions', type=str, required=True,
                        help='Path to the JSON file mapping KCs to question IDs')
    parser.add_argument('--json_file_cluster_desc', type=str, required=True,
                        help='Path to the JSON file mapping descs to question IDs')
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
    parser.add_argument('--eval_batch_size', type=int, default=128, help='Evaluation batch size')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Weighting factor for balancing KC-Quest and KC-Step contrastive loss components')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate for BERT model')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for the optimizer')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of epochs to train for')
    parser.add_argument('--patience', type=int, default=20, help='Patience for early stopping')
    parser.add_argument('--model_save_dir', type=str, default='xes3g5m',
                        help='Directory where the trained model will be saved')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Temperature scaling for softmax in the loss computation')
    parser.add_argument('--max_length', type=int, default=128, help='Maximum length for tokenized question text')
    parser.add_argument('--max_length_kc', type=int, default=32,
                        help='Maximum length for tokenized knowledge concept text')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA even if available')
    parser.add_argument('--disable_clusters', action='store_true',
                        help='Disable the false negatives elimination from clustering.')
    parser.add_argument('--wandb_project_name', type=str, default='default',required=False,
                        help='Name of the Wandb project to track the experiments.')

    args = parser.parse_args()
    main(args)
import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import argparse
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import BertModel, AutoModel, AutoTokenizer, BertTokenizer, BertConfig, BertModel, AutoModel, AutoTokenizer, RobertaTokenizer, RobertaConfig, get_linear_schedule_with_warmup
import os
import emoji
import re
from tqdm import tqdm
from utils import *

# Main runner script 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file", required=True, type=str, help="Path to fine-tuned model")
    parser.add_argument("--bert_model", default = "bert-base-uncased", required=True, type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--eval_file", required=True, type=str, help="CSV file with 'comment' and 'is_toxic'")
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--max_seq_length", default=128, type=int)    
    parser.add_argument("--output_file", required = True, type=str)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Starting Accuracy Evaluation")

    # Load Model & Tokenizer
    model, tokenizer = load_model(args.bert_model, args.model_file, device)
    print("Model Loaded")

    # Prepare dataset and dataloader
    data_loader = get_dataloader(tokenizer, args.eval_file, max_seq_length = args.max_seq_length, batch_size = args.batch_size) 
    print("Data Loaded")

    # Evaluate
    acc, fpr, fnr = evaluate(model, data_loader, device)
    print(f"Evaluation Accuracy: {acc*100:.2f}%")
    print(f"False Positive Rate: {fpr*100:.2f}%")
    print(f"False Negative Rate: {fnr*100:.2f}%")

    with open(args.output_file, "w") as f:
        f.write(f"Model File: {args.model_file}\n")
        f.write(f"Eval File: {args.eval_file}\n")
        f.write("\n===== Evaluation Metrics =====\n")
        f.write(f"Accuracy: {acc*100:.2f}%\n")
        f.write(f"False Positive Rate: {fpr*100:.2f}%\n")
        f.write(f"False Negative Rate: {fnr*100:.2f}%\n")

    print(f"\nSaved evaluation metrics to: {args.output_file}")
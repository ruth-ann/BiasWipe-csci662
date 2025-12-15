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
    parser.add_argument("--entity_terms_file", required=True, type=str, help=".txt file with line separated entity terms")
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--output_file", required = True, type=str)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load entity terms from a text file
    with open(args.entity_terms_file, "r", encoding="utf-8") as f:
        entity_terms = [line.strip() for line in f if line.strip()]
        
    # Load Model & Tokenizer
    model, tokenizer = load_model(args.bert_model, args.model_file, device)
    print("Model Loaded")

    # For EACH entity term, calculate accuracies
    accuracies = {}
    fprs = {}
    fnrs = {}
    lens = {}

    total_correct = 0
    total_count = 0

    # Open an output log file
    output_path = args.output_file
    with open(output_path, "w") as out:

        out.write(f"Model File: {args.model_file}\n")
        out.write(f"Eval File: {args.eval_file}\n")

        for term in entity_terms:
            print(f"\n\nENTITY TERM: {term}")
            out.write(f"\n\nENTITY TERM: {term}\n")

            # Prepare dataset + dataloader
            data_loader = get_dataloader(
                tokenizer, args.eval_file, entity_term=term,
                max_seq_length=args.max_seq_length,
                batch_size=args.batch_size
            )

            lens[term] = len(data_loader.dataset)
            print(f"Evaluating on {lens[term]} examples.")
            out.write(f"Evaluating on {lens[term]} examples.\n")

            # Evaluate
            acc, fpr, fnr = evaluate(model, data_loader, device)

            # Update running totals
            total_correct += acc * lens[term]
            total_count += lens[term]

            print(f"Evaluation Accuracy: {acc*100:.2f}%")
            print(f"False Positive Rate: {fpr*100:.2f}%")
            print(f"False Negative Rate: {fnr*100:.2f}%")

            out.write(f"Evaluation Accuracy: {acc*100:.2f}%\n")
            out.write(f"False Positive Rate: {fpr*100:.2f}%\n")
            out.write(f"False Negative Rate: {fnr*100:.2f}%\n")

            accuracies[term] = acc
            fprs[term] = fpr
            fnrs[term] = fnr

        # Weighted overall FPR/FNR
        total_examples = sum(lens[t] for t in entity_terms)
        overall_fpr = sum(fprs[t] * lens[t] for t in entity_terms) / total_examples
        overall_fnr = sum(fnrs[t] * lens[t] for t in entity_terms) / total_examples
        overall_acc = total_correct / total_count if total_count > 0 else 0

        print(f"\nWeighted Overall Accuracy: {overall_acc*100:.2f}%")
        print(f"Weighted Overall False Positive Rate: {overall_fpr*100:.2f}%")
        print(f"Weighted Overall False Negative Rate: {overall_fnr*100:.2f}%")

        out.write(f"\nWeighted Overall Accuracy: {overall_acc*100:.2f}%\n")
        out.write(f"Weighted Overall False Positive Rate: {overall_fpr*100:.2f}%\n")
        out.write(f"Weighted Overall False Negative Rate: {overall_fnr*100:.2f}%\n")

        # Equality metrics
        fped = sum(abs(overall_fpr - fprs[t]) for t in entity_terms)
        fned = sum(abs(overall_fnr - fnrs[t]) for t in entity_terms)

        print(f"\nFalse Positive Equality Difference (FPED): {fped:.4f}")
        print(f"False Negative Equality Difference (FNED): {fned:.4f}")

        out.write(f"\nFalse Positive Equality Difference (FPED): {fped:.4f}\n")
        out.write(f"False Negative Equality Difference (FNED): {fned:.4f}\n")

    print(f"\nSaved bias metrics to: {output_path}")

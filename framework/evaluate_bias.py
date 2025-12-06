import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import argparse
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import BertModel, AutoModel, AutoTokenizer, BertTokenizer, BertConfig
import os
import emoji
import re
from tqdm import tqdm

# Model Helpers
class OriginalBias(nn.Module):
    def __init__(self):
        super(OriginalBias, self).__init__()
        self.bert = AutoModel.from_pretrained(args.bert_model)
        self.linear2 = nn.Linear(768, 2) 

    def forward(self, input_ids,l):
        sequence_output= self.bert(input_ids,attention_mask=l)
        linear1_output = (sequence_output[0][:,0,:].view(-1,768)) 
        linear2_output = self.linear2(linear1_output)
        return linear2_output

# Dataset Helpers
def load_dataframe(filename):
    """ Load a pandas DataFrame from a .csv or .tsv file."""
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".csv":
        return pd.read_csv(filename, sep=",")
    elif ext == ".tsv":
        return pd.read_csv(filename, sep="\t")
    else:
        raise ValueError(f"Unsupported file type: {ext}. Use .csv or .tsv.")


def twitter_tokenizer(line):
    """ Preprocess tweet texts (normalizing casing, removing emojis, urls, usernames, noise)"""
    line = str(line)
    line = line.lower()
    line = emoji.demojize(line)
    line = re.sub(r'http\S+', ' ', line)
    line = re.sub('@[\w_]+', ' ', line)
    line = re.sub('\|LBR\|', '', line)
    line = re.sub('\.\.\.+', ' ', line)
    line = re.sub('!!+', '!', line)
    line = re.sub('\?\?+', '?', line)
    return line


def create_examples(df, entity_term):
    """ Create examples from a pandas dataframe """

    entity_data = df[df.keyword == entity_term].copy()

    texts = entity_data['comment'].apply(lambda x: twitter_tokenizer(x).strip())
    labels = entity_data['is_toxic']
    
    examples = list(zip(texts, labels))
    return examples


def covert_examples_to_tf(examples, max_seq_length, tokenizer):
    """ Convert examples list(elements are tuples of format (text, label) ) to BERT features """

    input_ids_list = []
    input_masks_list = []
    segment_ids_list = []
    label_ids_list = []

    for example in tqdm(examples, desc="Tokenizing Text Inputs"): 
        text, label_id = example 

        tokens = tokenizer.tokenize(text)

        # Truncate tokens to fit within max sequence length 
        if len(tokens) > max_seq_length - 2:
            tokens = tokens[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        input_ids_list.append(input_ids)
        input_masks_list.append(input_mask)
        segment_ids_list.append(segment_ids)
        label_ids_list.append(label_id)

    # Convert lists to pytorch 
    input_ids_list = torch.tensor(input_ids_list, dtype=torch.long)
    input_masks_list = torch.tensor(input_masks_list, dtype=torch.long)
    segment_ids_list = torch.tensor(segment_ids_list, dtype=torch.long)
    label_ids_list = torch.tensor(label_ids_list, dtype=torch.long)

    dataset = TensorDataset(input_ids_list, input_masks_list, segment_ids_list, label_ids_list)

    return dataset


def get_dataloader(tokenizer, filename, entity_term, max_seq_length = 120, batch_size = 16):
    df = load_dataframe(filename)
    examples = create_examples(df, entity_term)
    tf_dataset = covert_examples_to_tf(examples, max_seq_length, tokenizer)

    sampler = SequentialSampler(tf_dataset)
    dataloader = DataLoader(tf_dataset, sampler=sampler, batch_size=batch_size)

    return dataloader 

# Evaluation function 
def evaluate(model, data_loader, device, term):
    model.eval()
    correct = 0
    total = 0

    # Counters
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    with torch.no_grad():
        for input_ids, attention_mask, segment_ids, labels in tqdm(data_loader, desc=f"Evaluating {term}"):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model(input_ids, attention_mask)
            preds = 1 - torch.argmax(outputs, dim=1) # reversed outputs

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # Calculate TP, TN, FP, FN
            TP += ((preds == 1) & (labels == 1)).sum().item()
            TN += ((preds == 0) & (labels == 0)).sum().item()
            FP += ((preds == 1) & (labels == 0)).sum().item()
            FN += ((preds == 0) & (labels == 1)).sum().item()

    accuracy = correct / total
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
    fnr = FN / (FN + TP) if (FN + TP) > 0 else 0
    return accuracy, fpr, fnr


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

    entity_terms = [
        "lesbian",
        "gay",
        # "bisexual",
        # "transgender",
        "trans",
        # "queer",
        # "lgbt",
        # "lgbtq",
        "homosexual",
        # "straight",
        # "heterosexual",
        "male",
        "female",
        "nonbinary",
        # "african",
        # "african american",
        # "black",
        # "white",
        # "european",
        # "hispanic",
        # "latino",
        # "latina",
        # "latinx",
        # "mexican",
        # "canadian",
        # "american",
        # "asian",
        # "indian",
        # "middle eastern",
        # "chinese",
        # "japanese",
        # "christian",
        # "muslim",
        # "jewish",
        # "buddhist",
        # "catholic",
        # "protestant",
        # "sikh",
        # "taoist",
    ]

    # Load Model & Tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    config = BertConfig(num_labels = 2, num_hidden_layers=12) # CHECK FOR ACCURACY 
    model = OriginalBias()
    model.to(device)

    model_state_dict = torch.load(args.model_file, map_location=device)
    model.load_state_dict(model_state_dict)


    # For EACH entity term, 
    accuracies = {}
    fprs = {}
    fnrs = {}
    lens = {}

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
                tokenizer, args.eval_file, term,
                max_seq_length=args.max_seq_length,
                batch_size=args.batch_size
            )

            lens[term] = len(data_loader.dataset)
            print(f"Evaluating on {lens[term]} examples.")
            out.write(f"Evaluating on {lens[term]} examples.\n")

            # Evaluate
            acc, fpr, fnr = evaluate(model, data_loader, device, term)

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

        print(f"\nWeighted Overall False Positive Rate: {overall_fpr*100:.2f}%")
        print(f"Weighted Overall False Negative Rate: {overall_fnr*100:.2f}%")

        out.write(f"\nWeighted Overall False Positive Rate: {overall_fpr*100:.2f}%\n")
        out.write(f"Weighted Overall False Negative Rate: {overall_fnr*100:.2f}%\n")

        # Equality metrics
        fped = sum(abs(overall_fpr - fprs[t]) for t in entity_terms)
        fned = sum(abs(overall_fnr - fnrs[t]) for t in entity_terms)

        print(f"\nFalse Positive Equality Difference (FPED): {fped:.4f}")
        print(f"False Negative Equality Difference (FNED): {fned:.4f}")

        out.write(f"\nFalse Positive Equality Difference (FPED): {fped:.4f}\n")
        out.write(f"False Negative Equality Difference (FNED): {fned:.4f}\n")

    print(f"\nSaved bias metrics to: {output_path}")



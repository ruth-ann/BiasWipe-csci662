import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
import torch.nn as nn
import pandas as pd
import re, os, emoji
from tqdm import tqdm
from transformers import AutoModel, BertTokenizer, BertConfig, RobertaTokenizer, RobertaConfig


# Model Helpers
class OriginalBias(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.linear2 = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask):
        seq_out = self.bert(input_ids, attention_mask=attention_mask)
        pooled = seq_out[0][:,0,:]
        return self.linear2(pooled)

def load_model(model_name, model_file, device):
    # Set Up Model 
    model = OriginalBias(model_name)
    model.to(device)

    # Load Tokenizer
    if "roberta" in model_name:
        tokenizer = RobertaTokenizer.from_pretrained(model_name, do_lower_case=False)
        config = RobertaConfig(num_labels = 2, num_hidden_layers=12)
    elif "bert" in model_name:
        tokenizer = BertTokenizer.from_pretrained(model_name)
        config = BertConfig(num_labels = 2, num_hidden_layers=12) 
    else:
        raise ValueError(f"Unknown model type: {model}")

    # Load Model Weights 
    model_state_dict = torch.load(model_file, map_location=device)
    if "roberta" in model:
        new_state_dict = {}
        for k, v in model_state_dict.items():
            new_k = k.replace("roberta.", "bert.")
            new_state_dict[new_k] = v
        model_state_dict = new_state_dict
    model.load_state_dict(model_state_dict)
    
    return model, tokenizer

# Data Helpers 
def load_dataframe(filename):
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".csv":
        return pd.read_csv(filename)
    elif ext == ".tsv":
        return pd.read_csv(filename, sep="\t")
    else:
        raise ValueError("Unsupported file type")

def twitter_tokenizer(self, line):
    line = str(line)
    line = line.lower()
    line = emoji.demojize(line)
    line = re.sub(r'http\S+', ' ', line)
    line = re.sub(r'@[\w_]+', ' ', line)
    line = re.sub(r'\|LBR\|', '', line)
    line = re.sub(r'\.\.\.+', ' ', line)
    line = re.sub(r'!!+', '!', line)
    line = re.sub(r'\?\?+', '?', line)
    return line

def create_examples(df, entity_term=None):
    if entity_term is not None:
        df = df[df.keyword == entity_term]
    texts = df['comment'].apply(lambda x: twitter_tokenizer(x).strip())
    labels = df['is_toxic']
    return list(zip(texts, labels))

def covert_examples_to_tf(examples, max_seq_length, tokenizer, save_loc = None):
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

    if save_loc is not None:
        torch.save(dataset, save_loc)
    return dataset

def get_dataloader(tokenizer, filename, max_seq_length = 120, batch_size = 16, load_dataset = False):
    
    if load_dataset:
        tf_dataset = torch.load(filename, weights_only = False)
    else:
        df = load_dataframe(filename)
        examples = create_examples(df)
        tf_dataset = covert_examples_to_tf(examples, max_seq_length, tokenizer)

    sampler = SequentialSampler(tf_dataset)
    dataloader = DataLoader(tf_dataset, sampler=sampler, batch_size=batch_size)

    return dataloader 

# Evaluation Helpers
def evaluate(model, data_loader, device):
    model.eval()
    correct = total = 0

    # Counters for False / True   Positives / Negatives
    TP = TN = FP = FN = 0

    with torch.no_grad():
        for input_ids, attention_mask, segment_ids, labels in tqdm(data_loader, desc="Evaluating"):
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
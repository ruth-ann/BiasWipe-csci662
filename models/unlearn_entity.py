### IMPORTS ###
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import re
import pandas as pd
import emoji
import os
import logging
import argparse
import random
from tqdm import tqdm, trange
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time 
import torch
import copy
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

# from pytorch_pretrained_bert.tokenization import BertTokenizer
# from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from transformers import BertModel, AutoModel, AutoTokenizer, BertTokenizer, BertConfig, BertModel, AutoModel, AutoTokenizer, RobertaTokenizer, RobertaConfig, get_linear_schedule_with_warmup
from transformers import RobertaForSequenceClassification

from sklearn.metrics import accuracy_score,confusion_matrix,recall_score
from sklearn.metrics import classification_report


import os
import numpy as np
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

# Model Helper Functions
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


# Data Processing Helper Functions

def load_dataframe(filename):
    """ Load a pandas DataFrame from a .csv or .tsv file."""
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".csv":
        return pd.read_csv(filename, sep=",")
    elif ext == ".tsv":
        return pd.read_csv(filename, sep="\t")
    else:
        raise ValueError(f"Unsupported file type: {ext}. Use .csv or .tsv.")


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


def create_examples(df, entity_term, remove_keyword = False):
    """ Create examples from a pandas dataframe """

    entity_data = df[df.keyword == entity_term].copy()

    texts = entity_data['comment'].apply(lambda x: twitter_tokenizer(x).strip())

    if remove_keyword:
        pattern = rf"\b{entity_term}s?\b"
        entity_data['comment'] = entity_data['comment'].str.replace(pattern, '', regex=True)
    
    labels = entity_data['is_toxic']
    
    examples = list(zip(texts, labels))
    return examples




def covert_examples_to_tf(examples, max_seq_length, tokenizer):
    """ Convert examples list(elements are tuples of format (text, label) ) to BERT features """

    input_ids_list = []
    input_masks_list = []
    segment_ids_list = []
    label_ids_list = []

    for (ex_index, example) in enumerate(examples): 
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


def get_dataloader(tokenizer, filename, entity_term, remove_keyword = False, max_seq_length = 120, batch_size = 16):
    df = load_dataframe(filename)
    examples = create_examples(df, entity_term, remove_keyword)
    tf_dataset = covert_examples_to_tf(examples, max_seq_length, tokenizer)

    sampler = SequentialSampler(tf_dataset)
    dataloader = DataLoader(tf_dataset, sampler=sampler, batch_size=batch_size)

    return dataloader 


# Unlearning Helper Functions

def param_array_to_model(_model, arr, slist):
    model = copy.deepcopy(_model)
    
    if not arr.is_cuda and device.type == "cuda":
        print("Array was on CPU for no reason")
        arr = arr.to(device)

    # Split arr into parameter tensors according to slist
    start_index = 0
    param_list = []
    for shape in slist:
        end_index = start_index + np.prod(shape)
        param_tensor = arr[start_index:end_index].view(shape)
        param_list.append(param_tensor)
        start_index = end_index

    # Assign tensors to the new model's parameters
    with torch.no_grad():
        _index = 0
        for name, param in model.named_parameters():
            if "weight" in name or "bias" in name:
                param.copy_(param_list[_index])
                _index += 1

    return model


def model_to_param_array(model):
    param_list = []
    slist = []

    for param in model.parameters():
        slist.append(param.shape)
        param_list.append(param.data.view(-1))  

    arr = torch.cat(param_list)  
    arr = arr.to(device)         

    return arr, slist

def calculate_shapley_values_fa(model, data_loader, device, repeats=100):
    model_arr, model_slist = model_to_param_array(model)

    num_neurons = len(model_arr)
    shapley_values = torch.zeros(num_neurons, device = device)  # Initialize Shapley values for each neuron

    total_iterations = len(data_loader) * repeats  # total steps for tqdm
    pbar = tqdm(total=total_iterations, desc="Calculating Shapley Values")

    for input_ids, input_mask, segment_ids, label_ids in data_loader:

        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        for i in range(repeats):
            # Random permutation
            k = int(num_neurons * 0.25)
            perm = torch.randperm(num_neurons, device=device)[:k]
            
            # Zero neurons mask
            zeroed_neurons = torch.ones(num_neurons, device=device)  # put mask directly on GPU
            zeroed_neurons[perm] = 0
            zeroed_model_tensor = model_arr * zeroed_neurons
            zeroed_model = param_array_to_model(model, zeroed_model_tensor, model_slist)
            
            # Forward pass
            zeroed_output = zeroed_model(input_ids, input_mask)
            zeroed_output_soft = F.log_softmax(zeroed_output, dim=1)
            loss = F.cross_entropy(zeroed_output_soft, label_ids)
            loss.backward()
            
            # Gradient processing
            prev_index = 0
            index = 0
            for param in zeroed_model.parameters():
                prev_index = index
                index += len(param.flatten())
                if param.grad is not None:
                    grad = param.grad
                    shapley_values[prev_index:index] += torch.abs(grad.view(-1) * model_arr[prev_index:index])
            pbar.update(1)
    pbar.close()
    return shapley_values


def unlearning(model, forget, forget_neu, device, num_weights = 150, num_repetitions = 10, log_file = None):

    # Calculate Shapley Values
    tox_shapley_values = calculate_shapley_values_fa(model, forget, device, num_repetitions)  
    nontox_shapley_values = calculate_shapley_values_fa(model, forget_neu, device, num_repetitions)  

    # Calculate Shapley Value Differences 
    diff_shap_values_toxnontox = tox_shapley_values - nontox_shapley_values
    _, max_diff_shap_values_ind = torch.topk(diff_shap_values_toxnontox, num_weights)
    diff_shap_values = diff_shap_values_toxnontox[max_diff_shap_values_ind] 

    # Zero out model weights for top differences 
    model_arr, model_slist = model_to_param_array(model)
    model_arr[max_diff_shap_values_ind] = 0
    updated_model = param_array_to_model(model, model_arr, model_slist)  # Assign 'updated_model'

    # Write out information
    if log_file is not None:
        with open(log_file, "w") as f:

            f.write("Original Tox Shapley Values:\n")
            f.write("{}\n\n".format(tox_shapley_values))
            
            f.write("Original Non-Tox Shapley Values:\n")
            f.write("{}\n\n".format(nontox_shapley_values))
            
            f.write("Difference (Tox - Non-Tox):\n")
            f.write("{}\n\n".format(diff_shap_values_toxnontox))
            
            f.write(f"Top {num_weights} Differences:\n")
            f.write("{}\n".format(diff_shap_values))

    return updated_model  # Add a return statement to return the updated model



# Main running script
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model Information
    parser.add_argument("--bert_model", default = "bert-base-uncased", required=True, type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--fine_tuned_model_file", required= True, type=str)

    # Data Inputs
    parser.add_argument("--forget_file", required=True, type=str)
    parser.add_argument("--forget_neu_file", required=True, type=str)
    parser.add_argument("--entity_term", required=True, type=str)
    parser.add_argument("--num_weights", required=True, type=int)
    parser.add_argument("--num_repetitions", required=True, type=int)

    # Data Outputs 
    parser.add_argument("--model_output_file", required=True, type=str,
                        help="Where to save the unbiased model. For example, new_weights.bin")
    parser.add_argument("--log_file", default=None, type=str,
                        help="Where to save the unbiased model. For example, new_weights.bin")

    args = parser.parse_args()

    # Random Seed
    random.seed(67)
    np.random.seed(67)
    torch.manual_seed(67)

    # Prepare Cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    # Load Model
        # Load Model & Tokenizer
    if "roberta" in args.bert_model:
        tokenizer = RobertaTokenizer.from_pretrained(args.bert_model, do_lower_case=False)
        config = RobertaConfig(num_labels = 2, num_hidden_layers=12)
        
    elif "bert" in args.bert_model:
        tokenizer = BertTokenizer.from_pretrained(args.bert_model)
        config = BertConfig(num_labels = 2, num_hidden_layers=12)
        
    else:
        raise ValueError(f"Unknown model type: {args.bert_model}")

    model = OriginalBias()
    model.to(device)

    model_state_dict = torch.load(args.fine_tuned_model_file, map_location=device)

    if "roberta" in args.bert_model:
        new_state_dict = {}
        for k, v in model_state_dict.items():
            new_k = k.replace("roberta.", "bert.")
            new_state_dict[new_k] = v

        model_state_dict = new_state_dict

    model.load_state_dict(model_state_dict)
    print("Model Loaded")

    # Prepare Datasets
    forget_dataloader = get_dataloader(tokenizer, args.forget_file, args.entity_term)
    forget_neu_dataloader = get_dataloader(tokenizer, args.forget_neu_file, args.entity_term, remove_keyword = True)
    print("DataLoaders set up")
    print("Number of examples in forget_dataloader:", len(forget_dataloader.dataset))
    print("Number of examples in forget_neu_dataloader:", len(forget_neu_dataloader.dataset))


    # Do Unlearning 
    model = unlearning(model, forget_dataloader, forget_neu_dataloader, device, args.num_weights, args.num_repetitions, args.log_file)
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model itself
    torch.save(model_to_save.state_dict(), args.model_output_file)

    print(f"Unbiased {args.bert_model} model by {args.num_weights} weights on term {args.entity_term} saved to {args.model_output_file}")
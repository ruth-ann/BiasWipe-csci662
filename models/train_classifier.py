# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#	 http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

# from pytorch_pretrained_bert.tokenization import BertTokenizer
# from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from transformers import BertModel, AutoModel, AutoTokenizer, BertTokenizer, BertConfig, RobertaTokenizer, RobertaConfig, get_linear_schedule_with_warmup

from sklearn.metrics import accuracy_score,confusion_matrix,recall_score
from sklearn.metrics import classification_report


import os
import numpy as np
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]= '1'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
device = torch.device("cuda")

print("Hey There!")

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
					datefmt = '%m/%d/%Y %H:%M:%S',
					level = logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
	"""A single training/test example for simple sequence classification."""

	def __init__(self, guid, text_a, text_b=None, label=None):
		"""Constructs a InputExample.
		Args:
			guid: Unique id for the example.
			text_a: string. The untokenized text of the first sequence. For single
			sequence tasks, only this sequence must be specified.
			text_b: (Optional) string. The untokenized text of the second sequence.
			Only must be specified for sequence pair tasks.
			label: (Optional) string. The label of the example. This should be
			specified for train and dev examples, but not for test examples.
		"""
		self.guid = guid
		self.text_a = text_a
		self.text_b = text_b
		self.label = label


class InputFeatures(object):
	"""A single set of features of data."""

	def __init__(self, input_ids, input_mask, segment_ids, label_id):
		self.input_ids = input_ids
		self.input_mask = input_mask
		self.segment_ids = segment_ids
		self.label_id = label_id


class DataProcessor(object):
	"""Base class for data converters for sequence classification data sets."""

	def get_train_examples(self, train_tsv):
		"""Gets a collection of `InputExample`s for the train set."""
		raise NotImplementedError()

	def get_dev_examples(self, dev_tsv):
		"""Gets a collection of `InputExample`s for the dev set."""
		raise NotImplementedError()
	def get_test_examples(self, test_tsv):
		"""Gets a collection of `InputExample`s for the dev set."""
		raise NotImplementedError()

	
	def get_labels(self):
		"""Gets the list of labels for this data set."""
		raise NotImplementedError()

	@classmethod
	def _read_tsv(cls, input_file, quotechar=None):
		"""Reads a tab separated value file."""
		with open(input_file, "r", encoding='utf-8') as f:
			reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
			lines = []
			for line in reader:
				lines.append(line)
			return lines



class SentiProcessor(DataProcessor):
	"""Processor for Senti dataset."""

	def get_train_examples(self, train_tsv):
		"""See base class."""
		return self._create_examples(
			pd.read_csv(train_tsv,sep='\t'), "train")

	def get_dev_examples(self, dev_tsv):
		"""See base class."""
		return self._create_examples(
			pd.read_csv(dev_tsv,sep='\t'), "dev")

	def get_test_examples(self, test_tsv):
		"""See base class."""
		return self._create_examples(
			pd.read_csv(test_tsv,sep='\t'), "test")

	
	def get_labels(self):
		"""See base class."""
		return ["TRUE","FALSE"]

	def twitter_tokenizer(self, line):
		"""Preprocess the tweet texts"""
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

	def _create_examples(self, data, set_type):
		"""Creates examples for the training and dev sets."""
		examples = []
		for i in range(len(data)):
			guid = "%s-%s" % (set_type, i)
			text_a = self.twitter_tokenizer(data['comment'].loc[i])
			label = data['is_toxic'].loc[i]
			examples.append(
				InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
		return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
	"""Loads a data file into a list of `InputBatch`s."""

	label_map = {label : i for i, label in enumerate(label_list)}

	features = []
	for (ex_index, example) in enumerate(examples):
		tokens_a = tokenizer.tokenize(example.text_a)
		#logger.info("example:", example)
		tokens_b = None
		if example.text_b:
			tokens_b = tokenizer.tokenize(example.text_b)
			# Modifies `tokens_a` and `tokens_b` in place so that the total
			# length is less than the specified length.
			# Account for [CLS], [SEP], [SEP] with "- 3"
			_truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
		else:
			# Account for [CLS] and [SEP] with "- 2"
			if len(tokens_a) > max_seq_length - 2:
				tokens_a = tokens_a[:(max_seq_length - 2)]

		# The convention in BERT is:
		# (a) For sequence pairs:
		#  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
		#  type_ids: 0   0  0	0	0	 0	   0 0	1  1  1  1   1 1
		# (b) For single sequences:
		#  tokens:   [CLS] the dog is hairy . [SEP]
		#  type_ids: 0   0   0   0  0	 0 0
		#
		# Where "type_ids" are used to indicate whether this is the first
		# sequence or the second sequence. The embedding vectors for `type=0` and
		# `type=1` were learned during pre-training and are added to the wordpiece
		# embedding vector (and position vector). This is not *strictly* necessary
		# since the [SEP] token unambigiously separates the sequences, but it makes
		# it easier for the model to learn the concept of sequences.
		#
		# For classification tasks, the first vector (corresponding to [CLS]) is
		# used as as the "sentence vector". Note that this only makes sense because
		# the entire model is fine-tuned.
		tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
		segment_ids = [0] * len(tokens)

		if tokens_b:
			tokens += tokens_b + ["[SEP]"]
			segment_ids += [1] * (len(tokens_b) + 1)

		input_ids = tokenizer.convert_tokens_to_ids(tokens)

		# The mask has 1 for real tokens and 0 for padding tokens. Only real
		# tokens are attended to.
		input_mask = [1] * len(input_ids)

		# Zero-pad up to the sequence length.
		padding = [0] * (max_seq_length - len(input_ids))
		input_ids += padding
		input_mask += padding
		segment_ids += padding

		assert len(input_ids) == max_seq_length
		assert len(input_mask) == max_seq_length
		assert len(segment_ids) == max_seq_length
		#print("example.label:", example.label)	
		label_id = label_map[str(example.label).upper()]

		if ex_index < 5:
			logger.info("*** Example ***")
			logger.info("guid: %s" % (example.guid))
			logger.info("tokens: %s" % " ".join(
					[str(x) for x in tokens]))
			logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
			logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
			logger.info(
					"segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
			logger.info("label: %s (id = %d)" % (example.label, label_id))

		features.append(
				InputFeatures(input_ids=input_ids,
							  input_mask=input_mask,
							  segment_ids=segment_ids,
							  label_id=label_id))
	return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
	"""Truncates a sequence pair in place to the maximum length."""

	# This is a simple heuristic which will always truncate the longer sequence
	# one token at a time. This makes more sense than truncating an equal percent
	# of tokens from each, since if one sequence is very short then each token
	# that's truncated likely contains more information than a longer sequence.
	while True:
		total_length = len(tokens_a) + len(tokens_b)
		if total_length <= max_length:
			break
		if len(tokens_a) > len(tokens_b):
			tokens_a.pop()
		else:
			tokens_b.pop()

def accuracy(out, labels):
	outputs = np.argmax(out, axis=1)
	return np.sum(outputs == labels)

def warmup_linear(x, warmup=0.002):
	if x < warmup:
		return x/warmup
	return 1.0 - x


def train(model, train_dataloader, args):
	model.train()
	global_step = 0
	tr_loss = 0
	nb_tr_examples, nb_tr_steps = 0, 0
	for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
		batch = tuple(t.to(device) for t in batch)
		input_ids, input_mask, segment_ids, label_ids = batch
		inputs = {'input_ids': input_ids,
				'attention_mask': input_mask,
				'labels': label_ids}
		# outputs = model(**inputs)
		outputs = model(input_ids, input_mask)
		output=F.log_softmax(outputs, dim=1)
			
		criterion = nn.CrossEntropyLoss() ## If required define your own criterion
		loss=criterion(output, label_ids)

		# loss = outputs[0]
		if n_gpu > 1:
			loss = loss.mean() # mean() to average on multi-gpu.
		if args.gradient_accumulation_steps > 1:
			loss = loss / args.gradient_accumulation_steps

		if args.fp16:
			optimizer.backward(loss)
		else:
			loss.backward()

		tr_loss += loss.item()
		nb_tr_examples += input_ids.size(0)
		nb_tr_steps += 1
		if (step + 1) % args.gradient_accumulation_steps == 0:
			# modify learning rate with special warm up BERT uses
			lr_this_step = args.learning_rate * warmup_linear(global_step/t_total, args.warmup_proportion)
			for param_group in optimizer.param_groups:
				param_group['lr'] = lr_this_step
			optimizer.step()
			optimizer.zero_grad()
			global_step += 1
	epoch_loss = float(tr_loss/nb_tr_steps)
	# print(f'Epoch {z} loss --- {float(tr_loss/nb_tr_steps)}')
	return model, epoch_loss


def evaluate(model, eval_dataloader):
	model.eval()
	eval_loss, eval_accuracy = 0, 0
	nb_eval_steps, nb_eval_examples = 0, 0
	true_labels = []
	predicted_labels = []
	for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
		input_ids = input_ids.to(device)
		input_mask = input_mask.to(device)
		segment_ids = segment_ids.to(device)
		label_ids = label_ids.to(device)

		with torch.no_grad():
			inputs = {'input_ids': input_ids,
				  'attention_mask': input_mask,
				  'labels': label_ids}
			eval_outputs = model(input_ids, input_mask)
			output=F.log_softmax(eval_outputs, dim=1)
			
			criterion = nn.CrossEntropyLoss() ## If required define your own criterion
			loss=criterion(output, label_ids)

			# logits = model(input_ids, segment_ids, input_mask)
			logits = output
			tmp_eval_loss = loss
			# eval_outputs = model(**inputs)
			
			# logits = eval_outputs[1]
			# tmp_eval_loss = eval_outputs[0]

		predicted_labels = predicted_labels + torch.argmax(logits, dim=-1).tolist()
		true_labels = true_labels + label_ids.tolist()

		logits = logits.detach().cpu().numpy()
		label_ids = label_ids.to('cpu').numpy()
		tmp_eval_accuracy = accuracy(logits, label_ids)

		eval_loss += tmp_eval_loss.mean().item()
		eval_accuracy += tmp_eval_accuracy

		nb_eval_examples += input_ids.size(0)
		nb_eval_steps += 1

		print(true_labels)
		print(predicted_labels)
	eval_loss = eval_loss / nb_eval_steps
	eval_accuracy = eval_accuracy / nb_eval_examples
	return eval_loss, eval_accuracy, predicted_labels, true_labels

def testing(model, test_dataloader):
	'''
	sent="जातिवाद, वंशवाद, भ्रष्टाचार पर लग रहा विराम। नये सफर पर उत्तर प्रदेश"
	inputs = tokenizer.encode_plus(sent, add_special_tokens=True, return_tensors='pt')
	model(**inputs).logits.argmax().item()
	'''
	model.eval()
	test_loss, test_accuracy = 0, 0
	nb_test_steps, nb_test_examples = 0, 0
	test_true_labels = []
	test_predicted_labels = []
	for input_ids, input_mask, segment_ids, label_ids in tqdm(test_dataloader, desc="Evaluating"):
		input_ids = input_ids.to(device)
		input_mask = input_mask.to(device)
		segment_ids = segment_ids.to(device)
		label_ids = label_ids.to(device)

		with torch.no_grad():
			inputs = {'input_ids': input_ids,
				  'attention_mask': input_mask,
				  'labels': label_ids}
			# tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
			# eval_outputs = model(**inputs)
			outputs = model(input_ids, input_mask)
			print(outputs)
			logits=F.log_softmax(outputs, dim=1)
			# logits = eval_outputs[1]
			# tmp_eval_loss = eval_outputs[0]

		test_predicted_labels = test_predicted_labels + torch.argmax(logits, dim=-1).tolist()
		test_true_labels = test_true_labels + label_ids.tolist()

		logits = logits.detach().cpu().numpy()
		label_ids = label_ids.to('cpu').numpy()
		tmp_eval_accuracy = accuracy(logits, label_ids)

		test_loss += tmp_eval_loss.mean().item()
		test_accuracy += tmp_eval_accuracy

		nb_test_examples += input_ids.size(0)
		nb_test_steps += 1

	print(test_true_labels[0])
	print(test_predicted_labels[0])
	test_loss = test_loss / nb_test_steps
	test_accuracy = test_accuracy / nb_test_examples
	return test_loss, test_accuracy, test_predicted_labels, test_true_labels

class OriginalBias(nn.Module):
		def __init__(self):
			super(OriginalBias, self).__init__()
			self.model_name = args.bert_model
			if "roberta" in self.model_name:
				self.roberta = AutoModel.from_pretrained(args.bert_model)
			elif "bert" in self.model_name:
				self.bert = AutoModel.from_pretrained(args.bert_model)
			else:
				raise ValueError(f"Unknown Model Type {self.model_name}")
			
			#self.linear1 = nn.Linear(768, 256)
			self.linear2 = nn.Linear(768, 2) ## 3 is the number of classes in this example

		def forward(self, input_ids,l):
			if "roberta" in self.model_name:
				sequence_output= self.roberta(input_ids,attention_mask=l) #tensor=sequence_output[0]=torch.Size([16, 80, 768]), batch,len,dim
			elif "bert" in self.model_name:
				sequence_output= self.bert(input_ids,attention_mask=l) #tensor=sequence_output[0]=torch.Size([16, 80, 768]), batch,len,dim
			else:
				raise ValueError(f"Unknown Model Type {self.model_name}")
			
			linear1_output = (sequence_output[0][:,0,:].view(-1,768)) ## extract the 1st token's embeddings
			linear2_output = self.linear2(linear1_output)
			
			return linear2_output

if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	## Required parameters
	parser.add_argument("--train_tsv",
                    default=None,
                    type=str,
                    required=True,
                    help="Path to the training data .tsv file.")
	parser.add_argument("--dev_tsv",
						default=None,
						type=str,
						required=True,
						help="Path to the development/validation data .tsv file.")
	parser.add_argument("--test_tsv",
						default=None,
						type=str,
						required=True,
						help="Path to the test data .tsv file.")
	parser.add_argument("--bert_model", default=None, type=str, required=True,
						help="Bert pre-trained model selected in the list: bert-base-uncased, "
						"bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
						"bert-base-multilingual-cased, bert-base-chinese, dccuchile/bert-base-spanish-wwm-cased.")
	parser.add_argument("--task_name",
						default=None,
						type=str,
						required=True,
						help="The name of the task to train.")
	parser.add_argument("--output_dir",
						default=None,
						type=str,
						required=True,
						help="The output directory where the model predictions and checkpoints will be written.")

	## Other parameters
	parser.add_argument("--max_seq_length",
						default=128,
						type=int,
						help="The maximum total input sequence length after WordPiece tokenization. \n"
							 "Sequences longer than this will be truncated, and sequences shorter \n"
							 "than this will be padded.")
	parser.add_argument("--N",
						default=12,
						type=int,
						help="The number of stacked encoders. \n")
	parser.add_argument("--do_train",
						action='store_true',
						help="Whether to run training.")
	parser.add_argument("--do_eval",
						action='store_true',
						help="Whether to run eval on the dev set.")
	parser.add_argument("--do_lower_case",
						action='store_true',
						help="Set this flag if you are using an uncased model.")
	parser.add_argument("--train_batch_size",
						default=32,
						type=int,
						help="Total batch size for training.")
	parser.add_argument("--eval_batch_size",
						default=8,
						type=int,
						help="Total batch size for eval.")
	parser.add_argument("--learning_rate",
						default=5e-5,
						type=float,
						help="The initial learning rate for Adam.")
	parser.add_argument("--num_train_epochs",
						default=3.0,
						type=float,
						help="Total number of training epochs to perform.")
	parser.add_argument("--warmup_proportion",
						default=0.1,
						type=float,
						help="Proportion of training to perform linear learning rate warmup for. "
							 "E.g., 0.1 = 10%% of training.")
	parser.add_argument("--no_cuda",
						action='store_true',
						help="Whether not to use CUDA when available")
	parser.add_argument("--local_rank",
						type=int,
						default=-1,
						help="local_rank for distributed training on gpus")
	parser.add_argument('--seed',
						type=int,
						default=42,
						help="random seed for initialization")
	parser.add_argument('--gradient_accumulation_steps',
						type=int,
						default=1,
						help="Number of updates steps to accumulate before performing a backward/update pass.")
	parser.add_argument('--fp16',
						action='store_true',
						help="Whether to use 16-bit float precision instead of 32-bit")
	parser.add_argument('--loss_scale',
						type=float, default=0,
						help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
							 "0 (default value): dynamic loss scaling.\n"
							 "Positive power of 2: static loss scaling value.\n")

	args = parser.parse_args()

	processors = {
		
		"senti": SentiProcessor,
	}

	num_labels_task = {
		
		"senti": 2,
	}

	if args.local_rank == -1 or args.no_cuda:
		device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
		n_gpu = torch.cuda.device_count()
	else:
		torch.cuda.set_device(args.local_rank)
		device = torch.device("cuda", args.local_rank)
		n_gpu = 1
		# Initializes the distributed backend which will take care of sychronizing nodes/GPUs
		torch.distributed.init_process_group(backend='nccl')
	logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
		device, n_gpu, bool(args.local_rank != -1), args.fp16))

	if args.gradient_accumulation_steps < 1:
		raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
							args.gradient_accumulation_steps))

	args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if n_gpu > 0:
		torch.cuda.manual_seed_all(args.seed)

	if not args.do_train and not args.do_eval:
		raise ValueError("At least one of `do_train` or `do_eval` must be True.")

	# if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
	# 	raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
	# os.makedirs(args.output_dir, exist_ok=True)

	task_name = args.task_name.lower()

	if task_name not in processors:
		raise ValueError("Task not found: %s" % (task_name))

	processor = processors[task_name]()
	num_labels = num_labels_task[task_name]
	label_list = processor.get_labels()
	print("LABELS: ", label_list)

	if "roberta" in args.bert_model:
		RobertaTokenizer.from_pretrained(args.roberta_model, do_lower_case=args.do_lower_case)
	elif "bert" in args.bert_model:
		tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
	else:
		raise ValueError(f"Unknown Model Type {args.bert_model}")
	

	train_examples = None
	num_train_steps = None
	if args.do_train:
		train_examples = processor.get_train_examples(args.train_tsv)
		num_train_steps = int(
			len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)
		print("RHA: num examples:", len(train_examples))
	# Prepare model
	num_hidden_layers = args.N
	
	if "roberta" in args.bert_model:
		config = RobertaConfig(num_labels = num_labels, num_hidden_layers=num_hidden_layers)
	elif "bert" in args.bert_model:
		config = BertConfig(num_labels = num_labels, num_hidden_layers=num_hidden_layers)
	else:
		raise ValueError(f"Unknown Model Type {args.bert_model}")
	
	model=OriginalBias()
	if args.fp16:
		model.half()
	model.to(device)
	if args.local_rank != -1:
		try:
			from apex.parallel import DistributedDataParallel as DDP
		except ImportError:
			raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

		model = DDP(model)
	elif n_gpu > 1:
		model = torch.nn.DataParallel(model)

	# Prepare optimizer
	param_optimizer = list(model.named_parameters())
	no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
	optimizer_grouped_parameters = [
		{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
		{'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
		]
	t_total = num_train_steps
	if args.local_rank != -1:
		t_total = t_total // torch.distributed.get_world_size()
	if args.fp16:
		try:
			from apex.optimizers import FP16_Optimizer
			from apex.optimizers import FusedAdam
		except ImportError:
			raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

		optimizer = FusedAdam(optimizer_grouped_parameters,
							  lr=args.learning_rate,
							  bias_correction=False,
							  max_grad_norm=1.0)
		if args.loss_scale == 0:
			optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
		else:
			optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

	else:
		if "roberta" in args.bert_model:
			optimizer = optim.AdamW(
				optimizer_grouped_parameters,
				lr=args.learning_rate
			)

			scheduler = get_linear_schedule_with_warmup(
				optimizer,
				num_warmup_steps=int(args.warmup_proportion * t_total),
				num_training_steps=t_total
			)
		elif "bert" in args.bert_model:
			optimizer = BertAdam(optimizer_grouped_parameters,
							 lr=args.learning_rate,
							 warmup=args.warmup_proportion,
							 t_total=t_total)
		else:
			raise ValueError(f"Unknown Model Type {args.bert_model}")
		
		
	global_step = 0
	nb_tr_steps = 0
	tr_loss = 0
	if args.do_train:
		# Loading training data
		train_features = convert_examples_to_features(
			train_examples, label_list, args.max_seq_length, tokenizer)
		logger.info("***** Running training *****")
		logger.info("  Num examples = %d", len(train_examples))
		logger.info("  Batch size = %d", args.train_batch_size)
		logger.info("  Num steps = %d", num_train_steps)
		all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
		all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
		all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
		all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
		train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
		if args.local_rank == -1:
			train_sampler = RandomSampler(train_data)
		else:
			train_sampler = DistributedSampler(train_data)
		train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

		# Loading evaluation data
		eval_examples = processor.get_dev_examples(args.dev_tsv)
		eval_features = convert_examples_to_features(
			eval_examples, label_list, args.max_seq_length, tokenizer)
		logger.info("***** Running evaluation *****")
		logger.info("  Num examples = %d", len(eval_examples))
		logger.info("  Batch size = %d", args.eval_batch_size)
		all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
		all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
		all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
		all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
		eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

		# Run prediction for full data
		eval_sampler = SequentialSampler(eval_data)
		eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

		eval_loss_ = 1e5
		eval_accuracy_ = 0
		model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
		output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")

		for z in trange(int(args.num_train_epochs), desc="Epoch"):
			model, train_loss = train(model, train_dataloader, args)
			eval_loss, eval_accuracy, _, _ = evaluate(model, eval_dataloader)
			print(f'|Training loss: {train_loss} |Evaluation loss: {eval_loss} Evaluation Accuracy {eval_accuracy}|')
			if eval_accuracy > eval_accuracy_:
				eval_accuracy_ = eval_accuracy
				# model.save_pretrained(args.output_dir)
				# tokenizer.save_pretrained(args.output_dir)
				torch.save(model_to_save.state_dict(), output_model_file) # Save model on evaluation accuracy
		
		test_examples = processor.get_test_examples(args.test_tsv)
		test_features = convert_examples_to_features(
			test_examples, label_list, args.max_seq_length, tokenizer)
		logger.info("***** Running evaluation *****")
		logger.info("  Num examples = %d", len(test_examples))
		logger.info("  Batch size = %d", args.eval_batch_size)
		all_test_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
		all_test_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
		all_test_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
		all_test_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
		test_data = TensorDataset(all_test_input_ids, all_test_input_mask, all_test_segment_ids, all_test_label_ids)

		test_sampler = SequentialSampler(test_data)
		test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)

		# Load a trained model that you have fine-tuned
		model_state_dict = torch.load(output_model_file)
		model.load_state_dict(model_state_dict)
		model.to(device)

		label_names = ["TRUE", "FALSE"]
		label_dict = {i:x for i,x in enumerate(label_names)}

		eval_loss, eval_accuracy, predicted_labels, true_labels = evaluate(model, test_dataloader)
		print("label_names",label_dict, predicted_labels[0], true_labels[0])
		df = pd.DataFrame(columns=['text','actual_label','predicted_labels'])
		df['text'] = [i.text_a for i in test_examples]
		df['actual_label'] = [label_dict[i] for i in true_labels]
		df['predicted_labels'] = [label_dict[i] for i in predicted_labels]
		result = {'eval_loss': eval_loss,
				  'eval_accuracy': eval_accuracy,
				  'classification report\n': classification_report(true_labels, predicted_labels, target_names=label_names),
				  'lr':args.learning_rate,
				  'epochs': args.num_train_epochs,
				  'batch size': args.train_batch_size}
		print(result)
		

		output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
		with open(output_eval_file, "a") as writer:
			logger.info("***** Eval results *****")
			for key in sorted(result.keys()):
				logger.info("  %s = %s", key, str(result[key]))
				writer.write("%s = %s\n" % (key, str(result[key])))
		df.to_csv(os.path.join(args.output_dir, "hypothesis.csv"), sep=',')
		os.system(f'cat run_multi.sh>{os.path.join(args.output_dir, "experiment_setting.txt")}')
		from sklearn.metrics import f1_score,precision_score,recall_score
		print("confusion matrix", confusion_matrix(true_labels,predicted_labels))
		print("f1-macro",f1_score(true_labels, predicted_labels, average='macro'))
		print("f1-micro",f1_score(true_labels, predicted_labels, average='micro'))
		print("f1-weighted",f1_score(true_labels, predicted_labels, average='weighted'))

		print("precision-macro",precision_score(true_labels, predicted_labels, average='macro'))
		print("precision-micro",precision_score(true_labels, predicted_labels, average='micro'))
		print("precision-weighted",precision_score(true_labels, predicted_labels, average='weighted'))

		print("recall-macro",recall_score(true_labels, predicted_labels, average='macro'))
		print("recall-micro",recall_score(true_labels, predicted_labels, average='micro'))
		print("recall-weighted",recall_score(true_labels, predicted_labels, average='weighted'))

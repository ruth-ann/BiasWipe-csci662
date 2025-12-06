#!/bin/bash

# ------------------- Configuration ------------------- #
MODEL_NAME=old_bert
TOKENIZER_NAME=bert-base-multilingual-uncased
EVAL_FILE=/home/exouser/repos/BiasWipe-csci662/framework/train_es.tsv
BIAS_EVAL_FILE=/home/exouser/data/unintended-ml-bias-analysis/lydia_eval.csv
BATCH_SIZE=16
MAX_SEQ_LENGTH=128


# Evaluate Accuracy on ORIGINAL DATASET
# python evaluate_accuracy.py \
#   --model_file /home/exouser/models/pretrained/logs_$MODEL_NAME/pytorch_model.bin \
#   --bert_model bert-base-uncased \
#   --tokenizer_name "$TOKENIZER_NAME" \
#   --eval_file "$EVAL_FILE" \
#   --batch_size "$BATCH_SIZE" \
#   --max_seq_length "$MAX_SEQ_LENGTH"


# Evaluate Accuracy on ORIGINAL DATASET
python evaluate_bias.py \
  --model_file /home/exouser/models/pretrained/logs_$MODEL_NAME/pytorch_model.bin \
  --bert_model bert-base-uncased \
  --tokenizer_name "$TOKENIZER_NAME" \
  --eval_file "$BIAS_EVAL_FILE" \
  --batch_size "$BATCH_SIZE" \
  --max_seq_length "$MAX_SEQ_LENGTH"


# Evaluate Entity-Specific Accuracies (false positive/ false negative) & ratios

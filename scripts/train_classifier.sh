#!/bin/bash

TRAIN_TSV="/path/to/train.tsv"     # UPDATE ACCORDINGLY
DEV_TSV="/path/to/dev.tsv"         # UPDATE ACCORDINGLY
TEST_TSV="/path/to/test.tsv"       # UPDATE ACCORDINGLY

OUTPUT_DIR="/path/to/output/logs/" # UPDATE ACCORDINGLY

BERT_MODEL="dccuchile/bert-base-spanish-wwm-cased"
TASK_NAME="senti"

NUM_EPOCHS=3
LEARNING_RATE=1e-5
TRAIN_BATCH_SIZE=16
EVAL_BATCH_SIZE=8
MAX_SEQ_LENGTH=120
N=12

python models/train_classifier.py \
    --train_tsv $TRAIN_TSV \
    --dev_tsv $DEV_TSV \
    --test_tsv $TEST_TSV \
    --output_dir $OUTPUT_DIR \
    --bert_model $BERT_MODEL \
    --task_name $TASK_NAME \
    --num_train_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --max_seq_length $MAX_SEQ_LENGTH \
    --N $N \
    --do_train \
    --do_eval

echo "Training and evaluation complete. Logs and model saved to: $OUTPUT_DIR"

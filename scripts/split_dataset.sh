#!/bin/bash

INPUT_FILE="/path/to/dataset.tsv"   # UPDATE ACCORDINGLY 
OUTPUT_DIR="/path/to/splits/"      # UPDATE ACCORDINGLY 
TRAIN_SIZE=0.6
DEV_SIZE=0.2
TEST_SIZE=0.2

DATASET_NAME=dataset_name
TRAIN_OUT="$OUTPUT_DIR/${DATASET_NAME}_train.tsv"
DEV_OUT="$OUTPUT_DIR/${DATASET_NAME}_dev.tsv"
TEST_OUT="$OUTPUT_DIR/${DATASET_NAME}_test.tsv"

python split_dataset.py \
  --input "$INPUT_FILE" \
  --train_out "$TRAIN_OUT" \
  --dev_out "$DEV_OUT" \
  --test_out "$TEST_OUT" \
  --train_size "$TRAIN_SIZE" \
  --dev_size "$DEV_SIZE" \
  --test_size "$TEST_SIZE"

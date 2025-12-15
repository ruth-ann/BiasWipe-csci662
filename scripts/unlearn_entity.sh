#!/bin/bash

# Data
FORGET_FILE=/path/to/sentence_templates.csv       # Data containing sentence templates
FORGET_NEU_FILE=/path/to/sentence_templates.csv   # Data containing sentence templates

# Entity term to unlearn
ENTITY_TERM=gay

# Unlearning parameters
NUM_WEIGHTS=150
NUM_REPETITIONS=10

# Model
BERT_MODEL=roberta-base
PREV_MODEL_FILE=/path/to/model.bin
MODEL_OUTPUT_FILE="/path/to/unbiased_model.bin"  
LOG_FILE="/path/to/unlearning_log.txt"           

python models/unlearn_entity.py \
    --bert_model $BERT_MODEL \
    --fine_tuned_model_file "$PREV_MODEL_FILE" \
    --forget_file $FORGET_FILE \
    --forget_neu_file $FORGET_NEU_FILE \
    --entity_term "$ENTITY_TERM" \
    --num_weights $NUM_WEIGHTS \
    --num_repetitions $NUM_REPETITIONS \
    --model_output_file "$MODEL_OUTPUT_FILE" \
    --log_file "$LOG_FILE"
    
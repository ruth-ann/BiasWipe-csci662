#!/bin/bash

MODELS=(
    "pretrained/old_bert.bin"
    "unbiased/old_bert_lesbian_150.bin"
)

# ------------------- Configuration ------------------- #
MODEL_TYPE=bert-base-uncased
PRETRAINED_DIR=/home/exouser/models/pretrained
UNBIASED_DIR=/home/exouser/models/unbiased

EVAL_FILE=/home/exouser/repos/BiasWipe-csci662/framework/comments_with_labels.tsv
BIAS_EVAL_FILE=/home/exouser/data/unintended-ml-bias-analysis/english_entities.csv

OUTPUT_FOLDER=/home/exouser/data/outputs/evaluations

BATCH_SIZE=16
MAX_SEQ_LENGTH=128


for MODEL_REL_PATH in "${MODELS[@]}"; do
    # Determine the full path
    FOLDER=$(echo $MODEL_REL_PATH | cut -d'/' -f1)
    FILE=$(echo $MODEL_REL_PATH | cut -d'/' -f2)
    if [ "$FOLDER" == "pretrained" ]; then
        MODEL_PATH="$PRETRAINED_DIR/$FILE"
    else
        MODEL_PATH="$UNBIASED_DIR/$FILE"
    fi

    MODEL_NAME=$(basename "$MODEL_PATH" | sed 's/\..*$//')  # strip extension for output filenames

    echo "Evaluating model: $MODEL_NAME"

    # Accuracy Evaluation
    start_time=$(date +%s)
    python evaluate_accuracy.py \
        --model_file "$MODEL_PATH" \
        --bert_model "$MODEL_TYPE" \
        --eval_file "$EVAL_FILE" \
        --batch_size "$BATCH_SIZE" \
        --max_seq_length "$MAX_SEQ_LENGTH" \
        --output_file ${OUTPUT_FOLDER}/${MODEL_NAME}_accuracy_eval.txt
    end_time=$(date +%s)
    echo "evaluate_accuracy.py took $(($end_time - $start_time)) seconds"

    # Bias Evaluation
    start_time=$(date +%s)
    python evaluate_bias.py \
        --model_file "$MODEL_PATH" \
        --bert_model "$MODEL_TYPE" \
        --eval_file "$BIAS_EVAL_FILE" \
        --batch_size "$BATCH_SIZE" \
        --max_seq_length "$MAX_SEQ_LENGTH" \
        --output_file ${OUTPUT_FOLDER}/${MODEL_NAME}_bias_eval.txt
    end_time=$(date +%s)
    echo "evaluate_bias.py took $(($end_time - $start_time)) seconds"

    echo "-----------------------------------------"
done
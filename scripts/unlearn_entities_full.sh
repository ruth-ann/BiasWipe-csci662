#!/bin/bash

# =============================================================================
# Model Unlearning Script
# Usage: ./unlearn_entity.sh [bs|be|rs|re]
#   bs = BERT Spanish
#   be = BERT English
#   rs = RoBERTa Spanish
#   re = RoBERTa English
# =============================================================================

# Arguments
DATA_DIR=/home/exouser/data
MODEL_DIR=/home/exouser/models
TEMPLATE_DIR=/home/exouser/data/unintended-ml-bias-analysis


# Check for argument
if [ $# -eq 0 ]; then
    echo "Error: No argument provided"
    echo "Usage: $0 [bs|be|rs|re]"
    echo "  bs = BERT Spanish"
    echo "  be = BERT English"
    echo "  rs = RoBERTa Spanish"
    echo "  re = RoBERTa English"
    exit 1
fi

CONFIG=$1

# -----------------------------------------------------------------------------
# Configuration Based on Argument
# -----------------------------------------------------------------------------

case $CONFIG in
    bs)
        echo "Configuration: BERT Spanish"
        MODEL_NAME=bert_es
        BERT_MODEL=dccuchile/bert-base-spanish-wwm-cased
        PREV_MODEL_FILE=${MODEL_DIR}/ # UPDATE MODEL LOCATION
        TRAIN_FILE=${TEMPLATE_DIR}/spanish_entities.csv
        ENTITY_TERMS=("heterosexual" "judío" "lesbiana")
        ;;
    
    be)
        echo "Configuration: BERT English"
        MODEL_NAME=bert_en
        BERT_MODEL=bert-base-uncased
        PREV_MODEL_FILE=${MODEL_DIR}/ # UPDATE MODEL LOCATION
        TRAIN_FILE=${TEMPLATE_DIR}/english_entities.csv
        ENTITY_TERMS=("gay" "homosexual" "lesbian")
        ;;
    
    rs)
        echo "Configuration: RoBERTa Spanish"
        MODEL_NAME=roberta_es
        BERT_MODEL=bertin-project/bertin-roberta-base-spanish
        PREV_MODEL_FILE=${MODEL_DIR}/ # UPDATE MODEL LOCATION
        TRAIN_FILE=${TEMPLATE_DIR}/spanish_entities.csv
        ENTITY_TERMS=("bisexual" "homosexual" "heterosexual")
        ;;
    
    re)
        echo "Configuration: RoBERTa English"
        MODEL_NAME=roberta_en
        BERT_MODEL=roberta-base
        PREV_MODEL_FILE=${MODEL_DIR}/ # UPDATE MODEL LOCATION
        TRAIN_FILE=${TEMPLATE_DIR}/english_entities.csv
        ENTITY_TERMS=("gay" "homosexual" "queer")
        ;;
    
    *)
        echo "Error: Invalid argument '$CONFIG'"
        echo "Usage: $0 [bs|be|rs|re]"
        echo "  bs = BERT Spanish"
        echo "  be = BERT English"
        echo "  rs = RoBERTa Spanish"
        echo "  re = RoBERTa English"
        exit 1
        ;;
esac

# -----------------------------------------------------------------------------
# Common Configuration
# -----------------------------------------------------------------------------

NUM_WEIGHTS=100
NUM_REPETITIONS=10

OUTPUT_MODEL_DIR=${MODEL_DIR}/unbiased
OUTPUT_LOG_DIR=${DATA_DIR}/outputs/logs

# Create output directories if they don't exist
mkdir -p "$OUTPUT_MODEL_DIR"
mkdir -p "$OUTPUT_LOG_DIR"

# -----------------------------------------------------------------------------
# Unlearning Loop
# -----------------------------------------------------------------------------

echo ""
echo "Model Name: $MODEL_NAME"
echo "BERT Model: $BERT_MODEL"
echo "Initial Model: $PREV_MODEL_FILE"
echo "Train File: $TRAIN_FILE"
echo "Entity Terms: ${ENTITY_TERMS[*]}"
echo "Number of Weights: $NUM_WEIGHTS"
echo "Number of Repetitions: $NUM_REPETITIONS"
echo ""

UNLEARNED_TERMS=""  # will hold all terms concatenated

for ENTITY_TERM in "${ENTITY_TERMS[@]}"; do
    echo "========================================="
    echo "Unlearning entity: $ENTITY_TERM"
    echo "========================================="

    # Update the running list of unlearned terms
    if [ -z "$UNLEARNED_TERMS" ]; then
        UNLEARNED_TERMS="$ENTITY_TERM"
    else
        UNLEARNED_TERMS="${UNLEARNED_TERMS}_${ENTITY_TERM}"
    fi

    # Output paths
    MODEL_OUTPUT_FILE="${OUTPUT_MODEL_DIR}/${MODEL_NAME}_${UNLEARNED_TERMS}_${NUM_WEIGHTS}.bin"
    LOG_FILE="${OUTPUT_LOG_DIR}/${MODEL_NAME}_${UNLEARNED_TERMS}_${NUM_WEIGHTS}_shapley.txt"

    echo "Output Model: $MODEL_OUTPUT_FILE"
    echo "Log File: $LOG_FILE"
    echo ""

    # Run unlearning for this entity
    python unlearn_entity.py \
        --bert_model "$BERT_MODEL" \
        --fine_tuned_model_file "$PREV_MODEL_FILE" \
        --forget_file "${DATA_DIR}/$TRAIN_FILE" \
        --forget_neu_file "${DATA_DIR}/$TRAIN_FILE" \
        --entity_term "$ENTITY_TERM" \
        --num_weights $NUM_WEIGHTS \
        --num_repetitions $NUM_REPETITIONS \
        --model_output_file "$MODEL_OUTPUT_FILE" \
        --log_file "$LOG_FILE"
    
    # Use newly debiased model for the next iteration
    PREV_MODEL_FILE="$MODEL_OUTPUT_FILE"
    
    echo ""
done

echo "All unlearning complete!"
echo "Final model: $PREV_MODEL_FILE"
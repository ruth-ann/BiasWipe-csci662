MODEL_NAME=roberta_en  
MODEL_TYPE=roberta-base

NUM_WEIGHTS=100                                   # For consistent starting point
NUM_REPETITIONS=10
ENTITY_TERMS=("gay" "homosexual" "lesbian")

FORGET_FILE=/path/to/sentence_templates.csv       # Data containing sentence templates
FORGET_NEU_FILE=/path/to/sentence_templates.csv   # Data containing sentence templates

PREV_MODEL_FILE=/home/exouser/models/pretrained/logs_roberta_en/roberta.bin

MODEL_SAVE_DIR=/home/exouser/models/unbiased
LOG_SAVE_DIR=/home/exouser/data/outputs/logs
UNLEARNED_TERMS=""  # will hold all terms concatenated

for ENTITY_TERM in "${ENTITY_TERMS[@]}"
do
    echo "Unlearning entity: $ENTITY_TERM"

    # Update the running list of unlearned terms
    if [ -z "$UNLEARNED_TERMS" ]; then
        UNLEARNED_TERMS="$ENTITY_TERM"
    else
        UNLEARNED_TERMS="${UNLEARNED_TERMS}_${ENTITY_TERM}"
    fi

    # Output paths
    MODEL_OUTPUT_FILE="${MODEL_SAVE_DIR}/${MODEL_NAME}_${UNLEARNED_TERMS}_${NUM_WEIGHTS}.bin"
    LOG_FILE="{$LOG_SAVE_DIR}/${MODEL_NAME}_${UNLEARNED_TERMS}_${NUM_WEIGHTS}_shapley.txt"

    # Run unlearning for this entity
    python unlearn_entity.py \
      --bert_model $MODEL_TYPE \
      --fine_tuned_model_file "$PREV_MODEL_FILE" \
      --forget_file "$FORGET_FILE" \
      --forget_neu_file $FORGET_NEU_FILE \
      --entity_term "$ENTITY_TERM" \
      --num_weights $NUM_WEIGHTS \
      --num_repetitions $NUM_REPETITIONS \
      --model_output_file "$MODEL_OUTPUT_FILE" \
      --log_file "$LOG_FILE"
      
   # Use newly debiased model for the next iteration
    PREV_MODEL_FILE="$MODEL_OUTPUT_FILE"
done
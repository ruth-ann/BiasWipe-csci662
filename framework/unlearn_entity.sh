MODEL_NAME=roberta_en  
NUM_WEIGHTS=100 # for consistent starting point
NUM_REPETITIONS=10
TRAIN_FILE=unintended-ml-bias-analysis/english_entities.csv     

ENTITY_TERMS=("gay" "homosexual" "lesbian")

PREV_MODEL_FILE=/home/exouser/models/pretrained/logs_roberta_en/roberta.bin
# PREV_MODEL_FILE="/home/exouser/models/pretrained/logs_$MODEL_NAME/pytorch_model.bin"
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
    MODEL_OUTPUT_FILE="/home/exouser/models/unbiased/${MODEL_NAME}_${UNLEARNED_TERMS}_${NUM_WEIGHTS}.bin"
    LOG_FILE="/home/exouser/data/outputs/logs/${MODEL_NAME}_${UNLEARNED_TERMS}_${NUM_WEIGHTS}_shapley.txt"

    # Run unlearning for this entity
    python unlearn_entity.py \
      --bert_model roberta-base \
      --fine_tuned_model_file "$PREV_MODEL_FILE" \
      --forget_file "/home/exouser/data/$TRAIN_FILE" \
      --forget_neu_file "/home/exouser/data/$TRAIN_FILE" \
      --entity_term "$ENTITY_TERM" \
      --num_weights $NUM_WEIGHTS \
      --num_repetitions $NUM_REPETITIONS \
      --model_output_file "$MODEL_OUTPUT_FILE" \
      --log_file "$LOG_FILE"
      
   # Use newly debiased model for the next iteration
    PREV_MODEL_FILE="$MODEL_OUTPUT_FILE"
done


# python unlearn_entity.py \
#   --bert_model bert-base-uncased \
#   --fine_tuned_model_file /home/exouser/models/pretrained/logs_$MODEL_NAME/pytorch_model.bin \
#   --forget_file /home/exouser/data/$TRAIN_FILE \
#   --forget_neu_file /home/exouser/data/$TRAIN_FILE \
#   --entity_term $ENTITY_TERM \
#   --num_weights $NUM_WEIGHTS \
#   --model_output_file /home/exouser/models/unbiased/${MODEL_NAME}_${ENTITY_TERM}_${NUM_WEIGHTS}.bin \
#   --log_file /home/exouser/data/outputs/logs/${MODEL_NAME}_${ENTITY_TERM}_${NUM_WEIGHTS}_shapley.txt 
!/bin/bash

MODEL_NAME=bert_es_final
NUM_WEIGHTS=100        # for consistent starting point
NUM_REPETITIONS=10
TRAIN_FILE=unintended-ml-bias-analysis/english_entities.csv     

ENTITY_TERMS=("gay" "homosexual" "lesbian")

PREV_MODEL_FILE="/home/exouser/models/pretrained/logs_$MODEL_NAME/pytorch_model.bin"
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
    MODEL_OUTPUT_FILE="/home/exouser/models/unbiased/${MODEL_NAME}_${UNLEARNED_TERMS}_${NUM_WEIGHTS}.bin"
    LOG_FILE="/home/exouser/data/outputs/logs/${MODEL_NAME}_${UNLEARNED_TERMS}_${NUM_WEIGHTS}_shapley.txt"

    # Run unlearning for this entity (ROBERTA version)
    python unlearn_entity.py \
      --bert_model dccuchile/bert-base-spanish-wwm-cased \
      --fine_tuned_model_file "$PREV_MODEL_FILE" \
      --forget_file "/home/exouser/data/$TRAIN_FILE" \
      --forget_neu_file "/home/exouser/data/$TRAIN_FILE" \
      --entity_term "$ENTITY_TERM" \
      --num_weights $NUM_WEIGHTS \
      --num_repetitions $NUM_REPETITIONS \
      --model_output_file "$MODEL_OUTPUT_FILE" \
      --log_file "$LOG_FILE"

    # Use newly debiased model for the next iteration
    PREV_MODEL_FILE="$MODEL_OUTPUT_FILE"
done

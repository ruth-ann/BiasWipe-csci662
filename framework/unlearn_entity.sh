MODEL_NAME=old_bert
# ENTITY_TERM=gay       
NUM_WEIGHTS=150  
TRAIN_FILE=unintended-ml-bias-analysis/english_entities.csv     

ENTITY_TERMS=("gay" "homosexual" "lesbian")

PREV_MODEL_FILE="/home/exouser/models/pretrained/logs_$MODEL_NAME/pytorch_model.bin"

for ENTITY_TERM in "${ENTITY_TERMS[@]}"
do
    echo "Unlearning entity: $ENTITY_TERM"

    # Output paths
    MODEL_OUTPUT_FILE="/home/exouser/models/unbiased/${MODEL_NAME}_${ENTITY_TERM}_${NUM_WEIGHTS}.bin"
    LOG_FILE="/home/exouser/data/outputs/logs/${MODEL_NAME}_${ENTITY_TERM}_${NUM_WEIGHTS}_shapley.txt"

    # Run unlearning for this entity
    python unlearn_entity.py \
      --bert_model bert-base-uncased \
      --fine_tuned_model_file "$PREV_MODEL_FILE" \
      --forget_file "/home/exouser/data/$TRAIN_FILE" \
      --forget_neu_file "/home/exouser/data/$TRAIN_FILE" \
      --entity_term "$ENTITY_TERM" \
      --num_weights $NUM_WEIGHTS \
      --model_output_file "$MODEL_OUTPUT_FILE" \
      --log_file "$LOG_FILE"
    # uses newly debiased model from previous step for sequential unlearning
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

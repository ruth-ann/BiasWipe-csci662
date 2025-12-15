# Model checkpoint to evaluate
MODEL_PATH=/path/to/model_name.bin # UPDATE ACCORDINGLY 

# Pre-trained model type
MODEL_TYPE=roberta-base # UPDATE ACCORDINGLY 

# Evaluation files
EVAL_FILE=/path/to/sentence_templates.csv # UPDATE ACCORDINGLY
ENTITY_TERMS_FILE=/path/to/identity_terms.txt  # UPDATE ACCORDINGLY 

# Output file
EVAL_DIR=/path/to/dir
OUTPUT_FILE=$EVAL_DIR/model_bias_eval.txt # UPDATE ACCORDINGLY

# Evaluation parameters
BATCH_SIZE=32 
MAX_SEQ_LENGTH=128

# ------------------- Run Bias Evaluation ------------------- #
echo "Evaluating bias for model: $MODEL_PATH"
echo "Evaluation file: $EVAL_FILE"
echo "Entity terms file: $ENTITY_TERMS_FILE"

python evals/eval_bias.py \
    --model_file "$MODEL_PATH" \
    --bert_model "$MODEL_TYPE" \
    --eval_file "$EVAL_FILE" \
    --entity_terms_file "$ENTITY_TERMS_FILE" \
    --batch_size "$BATCH_SIZE" \
    --max_seq_length "$MAX_SEQ_LENGTH" \
    --output_file "$OUTPUT_FILE"

echo "Bias evaluation complete. Results saved to: $OUTPUT_FILE"

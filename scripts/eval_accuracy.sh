# Model checkpoint to evaluate
MODEL_PATH=/path/to/model_name.bin # UPDATE ACCORDINGLY

# Pre-trained model type
MODEL_TYPE=roberta-base # UPDATE ACCORDINGLY

# Evaluation file
EVAL_FILE=/path/to/sentence_templates.csv # UPDATE ACCORDINGLY

# Output file
EVAL_DIR=/path/to/dir
OUTPUT_FILE=$EVAL_DIR/${MODEL_NAME}_accuracy_eval.txt # UPDATE ACCORDINGLY

# Evaluation parameters
BATCH_SIZE=32
MAX_SEQ_LENGTH=128

# ------------------- Run Accuracy Evaluation ------------------- #
echo "Evaluating bias for model: $MODEL_PATH"
echo "Evaluation file: $EVAL_FILE"

python evals/evaluate_accuracy.py \
    --model_file "$MODEL_PATH" \
    --bert_model "$MODEL_TYPE" \
    --eval_file "$EVAL_FILE" \
    --batch_size "$BATCH_SIZE" \
    --max_seq_length "$MAX_SEQ_LENGTH" \
    --output_file "$OUTPUT_FILE"

echo "Accuracy evaluation complete. Results saved to: $OUTPUT_FILE"

MODELS=(
    # PRETRAINED MODELS 
    # /home/exouser/models/pretrained/logs_old_bert/bert.bin
    # /home/exouser/models/pretrained/logs_bert_es_final/bert_es.bin
    # /home/exouser/models/pretrained/logs_roberta_en/roberta.bin
    # /home/exouser/models/pretrained/logs_roberta_es_final/roberta_es.bin

    # /home/exouser/models/unbiased/bert_es_final_gay_100.bin
    /home/exouser/models/unbiased/roberta_en_gay_100.bin
    # "/home/exouser/models/pretrained/old_bert.bin" # removed for now since only doing ablation; uncomment for specific files
    # "unbiased/old_bert_lesbian_150.bin"
    # "/home/exouser/models/pretrained/logs_bert_es_final/pytorch_model.bin"
    # $(ls /home/exouser/models/unbiased/old_bert*.bin)
    # $(ls /home/exouser/models/unbiased/bert_es*.bin)
    # $(ls /home/exouser/models/unbiased/ablation/*.bin) # for ablation results to make figures 4 and 5
)

# ------------------- Configuration ------------------- #

### MODEL TYPES ###
# MODEL_TYPE=dccuchile/bert-base-spanish-wwm-cased
# MODEL_TYPE=bert-base-uncased
# MODEL_TYPE=bertin-project/bertin-roberta-base-spanish
MODEL_TYPE=roberta-base

### EVALUATION DATASETS ### 
# ENGLISH EVALUATION FILES
EVAL_FILE=/home/exouser/repos/BiasWipe-csci662/framework/dev.tsv
BIAS_EVAL_FILE=/home/exouser/data/unintended-ml-bias-analysis/english_entities.csv

# SPANISH EVALUATION FILES
# EVAL_FILE=/home/exouser/repos/BiasWipe-csci662/framework/dev_es.tsv
# BIAS_EVAL_FILE=/home/exouser/data/unintended-ml-bias-analysis/spanish_entities.csv

OUTPUT_FOLDER=/home/exouser/data/outputs/evaluations

BATCH_SIZE=32 # increasing for speed 
MAX_SEQ_LENGTH=128


for MODEL_PATH in "${MODELS[@]}"; do

    MODEL_NAME=$(basename "$MODEL_PATH" | sed 's/\..*$//')

    echo "Evaluating model: $MODEL_NAME"
    echo "Path: $MODEL_PATH"

    # ---- Accuracy Evaluation ---- #
    python evaluate_accuracy.py \
        --model_file "$MODEL_PATH" \
        --bert_model "$MODEL_TYPE" \
        --eval_file "$EVAL_FILE" \
        --batch_size "$BATCH_SIZE" \
        --max_seq_length "$MAX_SEQ_LENGTH" \
        --output_file ${OUTPUT_FOLDER}/${MODEL_NAME}_accuracy_eval.txt

    # # ---- Bias Evaluation ---- #
    # python evaluate_bias.py \
    #     --model_file "$MODEL_PATH" \
    #     --bert_model "$MODEL_TYPE" \
    #     --eval_file "$BIAS_EVAL_FILE" \
    #     --batch_size "$BATCH_SIZE" \
    #     --max_seq_length "$MAX_SEQ_LENGTH" \
    #     --output_file ${OUTPUT_FOLDER}/${MODEL_NAME}_bias_eval.txt

    echo "-----------------------------------------"
done
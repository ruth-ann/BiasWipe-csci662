#!/bin/bash

MODEL_NAME=roberta-en  
NUM_REPETITIONS=10
TRAIN_FILE=unintended-ml-bias-analysis/english_entities.csv     

FIXED_NUM_WEIGHTS=100                   
ABLATION_VALUES=(50 100 150 200 250)     
ENTITY_TERMS=("gay" "homosexual" "lesbian")

PRETRAINED_MODEL="/home/exouser/models/pretrained/logs_${MODEL_NAME}/pytorch_model.bin"
ABLATION_DIR="/home/exouser/models/unbiased/ablation"
LOG_DIR="/home/exouser/data/outputs/logs/ablation"
EVAL_DIR="/home/exouser/data/outputs/evaluation"
TEMPLATE_FILE="/home/exouser/data/template_dataset.csv"

mkdir -p "$ABLATION_DIR"
mkdir -p "$LOG_DIR"
mkdir -p "$EVAL_DIR"

#=============================================================================
# MAIN ABLATION LOOP
#=============================================================================

echo "==================================================================="
echo "ABLATION STUDY: BiasWipe Weight Pruning Analysis"
echo "==================================================================="
echo "Model: $MODEL_NAME"
echo "Steps 1-2: Fixed at ${FIXED_NUM_WEIGHTS} weights"
echo "Step 3 ablation values: ${ABLATION_VALUES[@]}"
echo "Starting pretrained model: $PRETRAINED_MODEL"
echo "==================================================================="
echo ""

# Track overall progress
TOTAL_RUNS=$((${#ABLATION_VALUES[@]} * 3))
CURRENT_RUN=0

for ABLATION_NUM_WEIGHTS in "${ABLATION_VALUES[@]}"
do
    echo ""
    echo "###################################################################"
    echo "# ABLATION RUN: Step 3 will use ${ABLATION_NUM_WEIGHTS} weights"
    echo "###################################################################"
    
    # Start fresh from original pretrained model
    PREV_MODEL_FILE="$PRETRAINED_MODEL"
    UNLEARNED_TERMS=""
    
    # Sequential unlearning: 3 steps
    for i in "${!ENTITY_TERMS[@]}"
    do
        ENTITY_TERM="${ENTITY_TERMS[$i]}"
        STEP_NUM=$((i + 1))
        CURRENT_RUN=$((CURRENT_RUN + 1))
        
        echo ""
        echo "-------------------------------------------------------------------"
        echo "Progress: Run ${CURRENT_RUN}/${TOTAL_RUNS}"
        echo "-------------------------------------------------------------------"
        
        # Determine number of weights for this step
        if [ $STEP_NUM -eq 3 ]; then
            NUM_WEIGHTS=$ABLATION_NUM_WEIGHTS
            echo "Step ${STEP_NUM}/3: Unlearning '${ENTITY_TERM}' [ABLATION: ${NUM_WEIGHTS} weights]"
        else
            NUM_WEIGHTS=$FIXED_NUM_WEIGHTS
            echo "Step ${STEP_NUM}/3: Unlearning '${ENTITY_TERM}' [FIXED: ${NUM_WEIGHTS} weights]"
        fi
        
        # Build cumulative name
        if [ -z "$UNLEARNED_TERMS" ]; then
            UNLEARNED_TERMS="$ENTITY_TERM"
        else
            UNLEARNED_TERMS="${UNLEARNED_TERMS}_${ENTITY_TERM}"
        fi
        
        # Define output paths
        if [ $STEP_NUM -eq 3 ]; then
            # For step 3, include ablation value in filename
            MODEL_SUFFIX="ablation${ABLATION_NUM_WEIGHTS}"
            MODEL_OUTPUT_FILE="${ABLATION_DIR}/${MODEL_NAME}_${UNLEARNED_TERMS}_${MODEL_SUFFIX}.bin"
            LOG_FILE="${LOG_DIR}/${MODEL_NAME}_${UNLEARNED_TERMS}_${MODEL_SUFFIX}_shapley.txt"
        else
            # For steps 1-2, use step number
            MODEL_OUTPUT_FILE="${ABLATION_DIR}/${MODEL_NAME}_${UNLEARNED_TERMS}_step${STEP_NUM}.bin"
            LOG_FILE="${LOG_DIR}/${MODEL_NAME}_${UNLEARNED_TERMS}_step${STEP_NUM}_shapley.txt"
        fi
        
        echo "Input model: $(basename $PREV_MODEL_FILE)"
        echo "Output model: $(basename $MODEL_OUTPUT_FILE)"
        
        # Run unlearning
        python unlearn_entity.py \
          --bert_model bert-base-uncased \
          --fine_tuned_model_file "$PREV_MODEL_FILE" \
          --forget_file "/home/exouser/data/$TRAIN_FILE" \
          --forget_neu_file "/home/exouser/data/$TRAIN_FILE" \
          --entity_term "$ENTITY_TERM" \
          --num_weights $NUM_WEIGHTS \
          --num_repetitions $NUM_REPETITIONS \
          --model_output_file "$MODEL_OUTPUT_FILE" \
          --log_file "$LOG_FILE"
        
        # Check success
        if [ $? -ne 0 ]; then
            echo "ERROR: Unlearning failed at step ${STEP_NUM} for entity '${ENTITY_TERM}'"
            exit 1
        fi
        
        echo "✓ Completed step ${STEP_NUM}"
        
        # Update for next step
        PREV_MODEL_FILE="$MODEL_OUTPUT_FILE"
    done
    
    echo ""
    echo ">>> Ablation run complete: ${ABLATION_NUM_WEIGHTS} weights for step 3"
    echo ">>> Final debiased model: $MODEL_OUTPUT_FILE"
    
    # Evaluate this ablation run
    echo ">>> Evaluating bias metrics..."
    EVAL_OUTPUT="${EVAL_DIR}/ablation_${ABLATION_NUM_WEIGHTS}_results.json"
    
    python evaluate_bias.py \
      --model_file "$MODEL_OUTPUT_FILE" \
      --template_file "$TEMPLATE_FILE" \
      --output_file "$EVAL_OUTPUT" \
      --metrics FPED FNED FPR accuracy
    
    if [ $? -eq 0 ]; then
        echo "✓ Evaluation saved to: $EVAL_OUTPUT"
    else
        echo "WARNING: Evaluation failed for ablation value ${ABLATION_NUM_WEIGHTS}"
    fi
done

#=============================================================================
# SUMMARY AND PLOTTING
#=============================================================================

echo ""
echo "==================================================================="
echo "ABLATION STUDY COMPLETE"
echo "==================================================================="
echo "Total runs completed: ${TOTAL_RUNS}"
echo ""
echo "Results location:"
echo "  Models: $ABLATION_DIR"
echo "  Logs: $LOG_DIR"
echo "  Evaluations: $EVAL_DIR"
echo ""
echo "To generate ablation plots (Figures 4 & 5):"
echo "  python plot_ablation.py --results_dir $EVAL_DIR --model_name $MODEL_NAME"
echo "==================================================================="

# Optionally auto-generate plots
read -p "Generate plots now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python plot_ablation.py --results_dir "$EVAL_DIR" --model_name "$MODEL_NAME"
fi
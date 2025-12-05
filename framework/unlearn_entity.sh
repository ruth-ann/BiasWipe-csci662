MODEL_NAME=old_bert
ENTITY_TERM=lesbian       
NUM_WEIGHTS=150             


python unlearn_entity.py \
  --bert_model bert-base-uncased \
  --fine_tuned_model_file /home/exouser/models/pretrained/logs_$MODEL_NAME/pytorch_model.bin \
  --forget_file /home/exouser/data/unintended-ml-bias-analysis/lydia_tester.csv \
  --forget_neu_file /home/exouser/data/unintended-ml-bias-analysis/lydia_tester.csv \
  --entity_term $ENTITY_TERM \
  --num_weights $NUM_WEIGHTS \
  --model_output_file /home/exouser/models/unbiased/${MODEL_NAME}_${ENTITY_TERM}_${NUM_WEIGHTS}.bin \
  --log_file /home/exouser/data/outputs/logs/${MODEL_NAME}_${ENTITY_TERM}_${NUM_WEIGHTS}_shapely.txt 

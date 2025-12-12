python3 train_classifier_roberta_es.py \
--num_train_epochs 3 \
--learning_rate 3e-6 \
--eval_batch_size 8 \
--roberta_model 'bertin-project/bertin-roberta-base-spanish' \
--data_dir '' \
--output_dir 'logs_roberta_es_final/' \
--task_name 'senti' \
--N 12 \
--train_batch_size 16 \
--max_seq_length 120 \
--do_eval \
--do_train


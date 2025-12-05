python3 unlearn_entity.py \
--num_train_epochs 3 \
--learning_rate 3e-5 \
--eval_batch_size 8 \
--bert_model 'bert-base-uncased' \
--data_dir 'data/' \
--output_dir 'logs_original/' \
--task_name 'senti' \
--N 12 \
--train_batch_size 16 \
--max_seq_length 120 \
--do_eval \
--do_train


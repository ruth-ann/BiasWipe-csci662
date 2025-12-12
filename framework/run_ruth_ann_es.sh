# python3 train_classifier_es.py \
# --num_train_epochs 3 \
# --learning_rate 1e-6 \
# --eval_batch_size 8 \
# --bert_model 'dccuchile/bert-base-spanish-wwm-cased' \
# --data_dir '' \
# --output_dir 'logs_bert_es/' \
# --task_name 'senti' \
# --N 12 \
# --train_batch_size 16 \
# --max_seq_length 120 \
# --do_eval \
# --do_train


# python3 train_classifier_es.py \
# --num_train_epochs 3 \
# --learning_rate 3e-6 \
# --eval_batch_size 8 \
# --bert_model 'dccuchile/bert-base-spanish-wwm-cased' \
# --data_dir '' \
# --output_dir 'logs_bert_es/' \
# --task_name 'senti' \
# --N 12 \
# --train_batch_size 16 \
# --max_seq_length 120 \
# --do_eval \
# --do_train


python3 train_classifier_es.py \
--num_train_epochs 3 \
--learning_rate 1e-5 \
--eval_batch_size 8 \
--bert_model 'dccuchile/bert-base-spanish-wwm-cased' \
--data_dir '' \
--output_dir 'logs_bert_es_final/' \
--task_name 'senti' \
--N 12 \
--train_batch_size 16 \
--max_seq_length 120 \
--do_eval \
--do_train

# python3 train_classifier_es.py \
# --num_train_epochs 3 \
# --learning_rate 3e-5 \
# --eval_batch_size 8 \
# --bert_model 'dccuchile/bert-base-spanish-wwm-cased' \
# --data_dir '' \
# --output_dir 'logs_bert_es/' \
# --task_name 'senti' \
# --N 12 \
# --train_batch_size 16 \
# --max_seq_length 120 \
# --do_eval \
# --do_train


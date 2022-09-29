# Searching entity using seq2seq structure
python train_search.py \
--do_train \
--do_eval \
--seed 42 \
--PLM /home/work/team08/model/kt-ulm-base \
--data_dir /home/work/team08/data_learn \
--train_data_file klue_ner_train_80.t \
--eval_data_file klue_ner_test_20.t \
--group_name Named_Entity_Search \
--num_train_epochs 3 \
--per_device_train_batch_size 64 \
--per_device_eval_batch_size 64 \
--max_input_length 128 \
--max_output_length 32 \
--save_strategy no \
--evaluation_strategy steps \
--eval_steps 1000 \
--logging_steps 100 \
--save_total_limit 5 \
--output_dir ./exps \
--logging_dir ./logs \
--learning_rate 5e-5 \
--weight_decay 1e-2 \
--generation_num_beams 1 \
--predict_with_generate True \
--generation_max_length 32 \
--warmup_ratio 0.05

## Predict
python predict_search.py \
--PLM /home/work/team08/kt-dev/exps/search \
--data_dir /home/work/team08/data_learn \
--eval_data_file klue_ner_test_20.t \
--per_device_eval_batch_size 64 \
--max_input_length 128 \
--generation_num_beams 1 \
--predict_with_generate True \
--generation_max_length 32 \
--output_dir ./results

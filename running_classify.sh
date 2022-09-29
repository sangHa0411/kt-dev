# Classifing entity using seq2seq structure
python train_classify.py \
--do_train \
--do_eval \
--seed 42 \
--PLM /home/work/team08/model/kt-ulm-base \
--data_dir /home/work/team08/data_learn \
--train_data_file klue_ner_train_80.t \
--eval_data_file klue_ner_test_20.t \
--group_name Named_Entity_Classify \
--num_train_epochs 3 \
--per_device_train_batch_size 32 \
--per_device_eval_batch_size 16 \
--max_input_length 128 \
--max_output_length 32 \
--save_strategy no \
--evaluation_strategy steps \
--eval_steps 500 \
--logging_steps 100 \
--save_total_limit 5 \
--output_dir ./exps \
--logging_dir ./logs \
--learning_rate 3e-5 \
--weight_decay 1e-2 \
--generation_num_beams 1 \
--predict_with_generate True \
--generation_max_length 32 \
--warmup_ratio 0.05

# NER
## Train
python train_ner.py \
--do_train \
--do_eval \
--seed 42 \
--PLM /home/work/team08/model/kt-ulm-base \
--data_dir /home/work/team08/data_learn \
--train_data_file klue_ner_train_80.t \
--eval_data_file klue_ner_test_20.t \
--group_name Named_Entity_Detect \
--overwrite_output_dir \
--warmup_ratio 0.05 \
--num_train_epochs 3 \
--per_device_train_batch_size 32 \
--per_device_eval_batch_size 16 \
--max_input_length 128 \
--save_strategy no \
--evaluation_strategy steps \
--eval_steps 500 \
--logging_steps 100 \
--output_dir ./exps \
--learning_rate 3e-5 \
--weight_decay 1e-2

## Predict
python predict_ner.py \
--PLM /home/work/team08/kt-dev/exps/ner \
--data_dir /home/work/team08/data_learn \
--eval_data_file klue_ner_test_20.t \
--per_device_eval_batch_size 16 \
--max_input_length 128 \
--output_dir ./results



# NER
## Train
python train_ner.py \
--do_train \
--do_eval \
--seed 42 \
--PLM /home/work/team08/model/kt-ulm-small \
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
--save_strategy steps \
--evaluation_strategy steps \
--save_steps 500 \
--eval_steps 500 \
--logging_steps 100 \
--save_total_limit 5 \
--load_best_model_at_end True \
--metric_for_best_model f1 \
--output_dir ./exps \
--learning_rate 5e-5 \
--weight_decay 1e-3 \

# ## Predict
# python predict_ner.py \
# --PLM ./exps/ner/fold-0 \
# --data_dir ./data \
# --data_file klue_ner_test.txt \
# --per_device_eval_batch_size 8 \
# --max_input_length 128 \
# --output_dir ./results
# NER
## Train
python train_ner.py \
--do_train \
--do_eval \
--seed 42 \
--group_name Named_Entity_Detect \
--overwrite_output_dir \
--max_steps 3000 \
--warmup_ratio 0.2 \
--fold_size 4 \
--num_train_epochs 3 \
--adam_beta1 0.9 \
--adam_beta1 0.98 \
--adam_epsilon 1e-9 \
--per_device_train_batch_size 16 \
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
import os
import json
import torch
import random
import numpy as np
import multiprocessing
from datasets import DatasetDict
from models.model import T5ForConditionalGeneration
from utils.metrics import Seq2SeqSearchMetrics, ScoreCalculator
from utils.loader import Loader
from utils.parser import Seq2SeqParser
from utils.seperate import Spliter
from utils.preprocessor import Seq2SeqSearchPreprocessor
from utils.encoder import Seq2SeqEncoder
from trainer import Seq2SeqTrainer

from arguments import ModelArguments, DataTrainingArguments, TrainingArguments, LoggingArguments

from transformers import (
    T5Config,
    T5TokenizerFast,
    HfArgumentParser,
    DataCollatorForSeq2Seq,
)

def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments, LoggingArguments)
    )
    model_args, data_args, training_args, logging_args = parser.parse_args_into_dataclasses()
    seed_everything(training_args.seed)

    # -- Loading datasets
    print("\nLoad datasets")
    train_data_loader = Loader(data_args.data_dir, data_args.train_data_file)
    train_dataset = train_data_loader.load(test_flag=False)
    
    eval_data_loader = Loader(data_args.data_dir, data_args.eval_data_file)
    eval_dataset = eval_data_loader.load(test_flag=True)

    # -- CPU counts
    cpu_cores = multiprocessing.cpu_count()
    num_proc = int(cpu_cores // 2)

    # -- Loading tokenizer
    print("\nLoad tokenizer")
    model_name = model_args.PLM
    tokenizer = T5TokenizerFast.from_pretrained(model_name, use_fast=True)

    output_dir = training_args.output_dir

    # -- Parsing datasets
    print("\nParse datasets")
    parser = Seq2SeqParser()
    train_dataset = parser(train_dataset)
    eval_dataset = parser(eval_dataset)

    eval_entities, eval_labels = [], []
    for d in eval_dataset :
        eval_entities.append(d["entities"])
        eval_labels.append(d["labels"])
    eval_examples = {"entities" : eval_entities, 'labels' : eval_labels}

    # -- Preprocessing datasets
    print("\nPreprocess datasets")
    tag_dict = {"QT" : "수량", "DT" : "날짜", "PS" : "사람", "LC" : "장소", "TI" : "시간", "OG" : "기관"}
    preprocessor = Seq2SeqSearchPreprocessor(tag_dict)
    train_dataset = preprocessor.preprocess4train(train_dataset, eval_flag=False)
    eval_dataset = preprocessor.preprocess4train(eval_dataset, eval_flag=True)
    datasets = DatasetDict({"train" : train_dataset, "validation" : eval_dataset})
    print(datasets)

    # -- Encoding datasets
    print("\nEncode datasets")
    encoder = Seq2SeqEncoder(tokenizer, data_args.max_input_length, data_args.max_output_length)
    datasets = datasets.map(encoder, batched=True, num_proc=num_proc)
    datasets = datasets.remove_columns(["inputs"])
    print(datasets)

    # -- Loading config & Model
    print("\nLoading Model")
    config = T5Config.from_pretrained(model_args.PLM)
    model = T5ForConditionalGeneration.from_pretrained(model_args.PLM, config=config)

    # -- DataCollator
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # -- Metrics
    tag_list = list(tag_dict.keys())
    score_calculator = ScoreCalculator()
    metrics = Seq2SeqSearchMetrics(tokenizer, score_calculator, eval_examples, tag_list)
    compute_metrics = metrics.compute_metrics

    # -- Trainer
    target_dir = os.path.join(output_dir, "search")
    if not os.path.exists(target_dir) :
        os.mkdir(target_dir)

    training_args.output_dir = target_dir
    training_args.dataloader_num_workers = num_proc
    trainer = Seq2SeqTrainer(
        model,
        training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # -- Training
    if training_args.do_train :
        print("\nTraining")
        trainer.train()

    # -- Evaluation
    if training_args.do_eval :
        print("\nEvaluating")
        eval_metrics = trainer.evaluate()
        print(eval_metrics)

    trainer.save_model(target_dir)


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    np.random.default_rng(seed)
    random.seed(seed)


if __name__ == "__main__":
    main()
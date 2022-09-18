import os
import json
import torch
import random
import numpy as np
import wandb
import multiprocessing
from dotenv import load_dotenv
from datasets import DatasetDict
from models.metrics import Metrics
from models.model import T5ForConditionalGeneration
from utils.loader import Loader
from utils.parser import parsing
from utils.seperate import Spliter
from utils.preprocessor import Preprocessor
from utils.encoder import Encoder

from arguments import ModelArguments, DataTrainingArguments, TrainingArguments, LoggingArguments

from transformers import (
    T5Config,
    T5TokenizerFast,
    HfArgumentParser,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer
)

def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments, LoggingArguments)
    )
    model_args, data_args, training_args, logging_args = parser.parse_args_into_dataclasses()
    seed_everything(training_args.seed)

    # -- Loading datasets
    print("\nLoad datasets")
    loader = Loader(data_args.data_dir, data_args.data_file)
    raw_dataset = loader.load()

    # -- CPU counts
    cpu_cores = multiprocessing.cpu_count()
    num_proc = int(cpu_cores // 2)

    # -- Loading tokenizer
    print("\nLoad tokenizer")
    model_name = model_args.PLM
    tokenizer = T5TokenizerFast.from_pretrained(model_name, use_fast=True)
    
    # -- Spliting datasets
    print("\nSeperate datasets")
    spliter = Spliter(raw_dataset, training_args.fold_size)

    output_dir = training_args.output_dir
    load_dotenv(dotenv_path=logging_args.dotenv_path)
    # -- K Fold Training
    for i in range(training_args.fold_size) :
        i = 4
        print("\n%dth Training" %i)
        train_ids, validation_ids = spliter.get_dataset(i)

        # Parsing datasets
        print("\nParse datasets")
        dataset = parsing(raw_dataset)
        print(dataset)

        train_dataset = dataset.select(train_ids)
        validation_dataset = dataset.select(validation_ids)    

        datasets = DatasetDict({"train" : train_dataset, "validation" : validation_dataset})
        print(dataset)

        # Preprocessing datasets
        print("\nPreprocess datasets")
        preprocessor = Preprocessor(tokenizer)
        datasets = datasets.map(preprocessor, batched=True, num_proc=num_proc)
        datasets = datasets.remove_columns(["sentences", "entities"])
        print(datasets)

        # Encoding datasets
        print("\nEncode datasets")
        encoder = Encoder(tokenizer, data_args.max_input_length, data_args.max_output_length)
        datasets = datasets.map(encoder, batched=True, num_proc=num_proc)
        datasets = datasets.remove_columns(["inputs"])
        print(datasets)

        # Loading config & Model
        print("\nLoading Model")
        config = T5Config.from_pretrained(model_args.PLM)
        model = T5ForConditionalGeneration.from_pretrained(model_args.PLM, config=config)

        # DataCollator
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

        # Metrics
        metrics = Metrics(tokenizer)
        compute_metrics = metrics.compute_metrics

        # Trainer
        target_dir = os.path.join(output_dir, f"fold-{i}")
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

        WANDB_AUTH_KEY = os.getenv('WANDB_AUTH_KEY')
        wandb.login(key=WANDB_AUTH_KEY)

        args = training_args
        wandb_name = f"EP:{args.num_train_epochs}_BS:{args.per_device_train_batch_size}_LR:{args.learning_rate}_WD:{args.weight_decay}_WR:{args.warmup_ratio}_FOLD:{i}"
        wandb.init(
            entity="sangha0411",
            project=logging_args.project_name, 
            name=wandb_name,
            group=logging_args.group_name)
        wandb.config.update(training_args)

        # Training
        if training_args.do_train :
            print("\nTraining")
            trainer.train()

        # Evaluation
        if training_args.do_eval :
            print("\nEvaluating")
            trainer.evaluate()

        wandb.finish()
        break


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
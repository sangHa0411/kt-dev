import os
import json
import torch
import random
import numpy as np
import importlib
import copy
import multiprocessing
from datasets import DatasetDict
from models.metrics import Metrics
from utils.loader import Loader
from utils.parser import parsing
from utils.seperate import Spliter
from utils.preprocessor import Preprocessor
from utils.encoder import Encoder

from arguments import ModelArguments, DataTrainingArguments, TrainingArguments, LoggingArguments

from transformers import (
    T5Config,
    T5TokenizerFast,
    T5ForConditionalGeneration,
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
    special_tokens_dict = {'additional_special_tokens': ['<QT>', '<DT>', '<PS>', '<LC>', '<TI>', '<OG>']}
    sep_token = {'sep_token': '<sep>'}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    sep_tok = tokenizer.add_special_tokens(sep_token)
    print(tokenizer)

    # -- Spliting datasets
    print("\nSeperate datasets")
    spliter = Spliter(raw_dataset, training_args.fold_size)

    # -- K Fold Training
    for i in range(training_args.fold_size) :
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
        config = T5Config.from_pretrained(model_args.PLM)
        model = T5ForConditionalGeneration.from_pretrained(model_args.PLM, config=config)

        # DataCollator
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

        # Trainer
        trainer = Seq2SeqTrainer(
            model,
            training_args,
            train_dataset=datasets["train"],
            eval_dataset=datasets["validation"],
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=Metrics.compute_metrics
        )

        # Training
        trainer.train()

        # Evaluation
        trainer.evaluate()




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
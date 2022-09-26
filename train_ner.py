import os
import json
import torch
import copy
import random
import numpy as np
import wandb
import transformers
import multiprocessing
from dotenv import load_dotenv
from datasets import DatasetDict
from models.model import T5EncoderModel
from utils.metrics import NERMetrics, ScoreCalculator
from utils.loader import Loader
from utils.parser import NERParser
from utils.postprocessor import NERPostprocessor
from utils.encoder import NEREncoder
from trainer import Trainer

from arguments import ModelArguments, DataTrainingArguments, TrainingArguments, LoggingArguments

from transformers import (
    T5Config,
    T5TokenizerFast,
    HfArgumentParser,
    DataCollatorForTokenClassification,
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
 
    # -- Parsing datasets
    print("\nParse datasets")   
    parser = NERParser()
    eval_tag_words, eval_tag_names = parser.extract(eval_dataset)
    train_dataset = parser(train_dataset)
    eval_dataset = parser(eval_dataset)    
    eval_sentences = copy.deepcopy(eval_dataset["sentences"])
    eval_examples = {"sentences" : eval_sentences,
        "tag_words" : eval_tag_words,
        "tag_names" : eval_tag_names
    }
    
    datasets = DatasetDict({"train" : train_dataset, "validation" : eval_dataset})
    print(datasets)

    # -- CPU counts
    cpu_cores = multiprocessing.cpu_count()
    num_proc = int(cpu_cores // 2)

    # -- Loading tokenizer
    print("\nLoad tokenizer")
    model_name = model_args.PLM
    tokenizer = T5TokenizerFast.from_pretrained(model_name, use_fast=True)
    
    # -- Encoder
    label_dict = {"O" : 0, "QT" : 1, "DT" : 2, "PS" : 3, "LC" : 4, "TI" : 5, "OG" : 6}
    encoder = NEREncoder(tokenizer, data_args.max_input_length, label_dict)

    eval_dataset_ = eval_dataset.map(encoder, batched=True, num_proc=num_proc)

    # -- Encoding datasets
    print("\nEncode datasets")
    datasets = datasets.map(encoder, batched=True, num_proc=num_proc)
    datasets = datasets.remove_columns(["sentences"])
    print(datasets)

    output_dir = training_args.output_dir
    load_dotenv(dotenv_path=logging_args.dotenv_path)

    # Loading config & Model
    print("\nLoading Model")
    config = T5Config.from_pretrained(model_args.PLM)
    config.num_labels = len(label_dict)
    model = T5EncoderModel.from_pretrained(model_args.PLM, config=config)

    # DataCollator
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, 
        padding=True,
        max_length=data_args.max_input_length   
    )

    # Metrics
    score_calculator = ScoreCalculator()
    metrics = NERMetrics()
    compute_metrics = metrics.compute_metrics

    # Trainer
    checkpoint_dir = os.path.join(training_args.output_dir, "ner")
    if not os.path.exists(checkpoint_dir) :
        os.mkdir(checkpoint_dir)

    training_args.output_dir = checkpoint_dir
    training_args.dataloader_num_workers = num_proc

    # Postprocessor
    inverse_label_dict = {i : t for t, i in label_dict.items()}
    postprocessor = NERPostprocessor(tokenizer, data_args.max_input_length, inverse_label_dict)
    trainer = Trainer(
        model,
        training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        eval_examples=eval_examples,
        score_calculator=score_calculator,
        postprocessor=postprocessor,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    WANDB_AUTH_KEY = os.getenv('WANDB_AUTH_KEY')
    wandb.login(key=WANDB_AUTH_KEY)

    args = training_args
    wandb_name = f"EP:{args.num_train_epochs}_BS:{args.per_device_train_batch_size}_LR:{args.learning_rate}_WD:{args.weight_decay}_WR:{args.warmup_ratio}"
    wandb.init(
        entity="sangha0411",
        project=logging_args.project_name, 
        name=wandb_name,
        group=logging_args.group_name
    )
    wandb.config.update(training_args)

    # Training
    if training_args.do_train :
        print("\nTraining")
        trainer.train()

    # Evaluation
    if training_args.do_eval :
        print("\nEvaluating")
        eval_metrics = trainer.evaluate()
        print(eval_metrics)

    # trainer.save_model(checkpoint_dir)
    wandb.finish()


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
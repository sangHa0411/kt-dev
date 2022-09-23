import os
import json
from unittest import result
import torch
import pandas as pd 
import numpy as np
import multiprocessing
from datasets import Dataset
from models.model import T5EncoderModel
from utils.loader import Loader
from utils.postprocessor import NERPostprocessor
from utils.encoder import NEREncoder

from arguments import ModelArguments, DataTrainingArguments, TrainingArguments, LoggingArguments

from transformers import (
    T5Config,
    T5TokenizerFast,
    HfArgumentParser,
    DataCollatorForTokenClassification,
    Trainer,
)


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments, LoggingArguments)
    )
    model_args, data_args, training_args, logging_args = parser.parse_args_into_dataclasses()

    # -- Loading datasets
    print("\nLoad datasets")
    loader = Loader(data_args.data_dir, data_args.data_file)
    raw_dataset = loader.load(test_flag=False)

    raw_df = pd.DataFrame({"sentences" : raw_dataset})
    dataset = Dataset.from_pandas(raw_df)
    print(dataset)

    # -- CPU counts
    cpu_cores = multiprocessing.cpu_count()
    num_proc = int(cpu_cores // 2)

    # -- Loading tokenizer
    print("\nLoad tokenizer")
    model_name = model_args.PLM
    tokenizer = T5TokenizerFast.from_pretrained(model_name, use_fast=True)
    
    # -- Encoder
    print("\nEncoding Datasets")
    encoder = NEREncoder(tokenizer, data_args.max_input_length)
    dataset = dataset.map(encoder, batched=True, num_proc=num_proc)
    dataset = dataset.remove_columns(["sentences"])

    # Loading config & Model
    print("\nLoading Model")
    config = T5Config.from_pretrained(model_args.PLM)
    model = T5EncoderModel.from_pretrained(model_args.PLM, config=config)

    # DataCollator
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, 
        padding=True,
        max_length=data_args.max_input_length   
    )

    # Inference
    training_args.dataloader_num_workers = num_proc
    trainer = Trainer(
        model,
        training_args,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    predictions = trainer.predict(test_dataset=dataset)
    
    # Postprocessing
    tag_dict = {0 : "O", 1 : "QT", 2 : "DT", 3 : "PS" ,4  :  "LC", 5 : "TI", 6 : "OG" }
    postprocessor = NERPostprocessor(tokenizer, data_args.max_input_length, tag_dict)
    word_list, tag_list = postprocessor(predictions[0], raw_dataset)
    breakpoint()

if __name__ == "__main__":
    main()
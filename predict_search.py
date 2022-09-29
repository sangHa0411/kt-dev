import os
import torch
import random
import pandas as pd
import numpy as np
import multiprocessing
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.model import T5ForConditionalGeneration
from utils.metrics import Seq2SeqSearchMetrics, ScoreCalculator
from utils.loader import Loader
from utils.collator import DataCollatorForSeq2Seq
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
)

OUTPUT_FILE = "prediction.csv"

def preprocess(array, tokenizer) :
    eos_token_id = tokenizer.eos_token_id

    index = len(array) - 1
    if eos_token_id in array :
        index = array.index(eos_token_id)

    valid = array[:index]
    return valid


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments, LoggingArguments)
    )
    model_args, data_args, training_args, logging_args = parser.parse_args_into_dataclasses()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -- Loading datasets
    print("\nLoad datasets")
    data_loader = Loader(data_args.data_dir, data_args.eval_data_file)
    test_dataset = data_loader.load(test_flag=True)

    # -- Parsing datasets
    print("\nParse datasets")   
    parser = Seq2SeqParser()
    test_dataset = parser(test_dataset)
    raw_sentences = [d["sentences"] for d in test_dataset]
    
    # -- Formatting dataset from datafream
    raw_df = pd.DataFrame({"sentences" : raw_sentences})
    test_dataset = Dataset.from_pandas(raw_df)
    print(test_dataset)

    # -- CPU counts
    cpu_cores = multiprocessing.cpu_count()
    num_proc = int(cpu_cores // 2)

    # -- Loading tokenizer
    print("\nLoad tokenizer")
    model_name = model_args.PLM
    tokenizer = T5TokenizerFast.from_pretrained(model_name, use_fast=True)

    # -- Preprocessing datasets
    print("\nPreprocess datasets")
    tag_dict = {"QT" : "수량", "DT" : "날짜", "PS" : "사람", "LC" : "장소", "TI" : "시간", "OG" : "기관"}
    preprocessor = Seq2SeqSearchPreprocessor(tag_dict)
    test_dataset = preprocessor.preprocess4test(test_dataset)

    # -- Encoding datasets
    print("\nEncode datasets")
    encoder = Seq2SeqEncoder(tokenizer, data_args.max_input_length, data_args.max_output_length)
    test_dataset = test_dataset.map(encoder, batched=True, num_proc=num_proc)
    test_dataset = test_dataset.remove_columns(["inputs"])
    print(test_dataset)

    # Loading config & Model
    print("\nLoading Model")
    config = T5Config.from_pretrained(model_args.PLM)
    model = T5ForConditionalGeneration.from_pretrained(model_args.PLM, config=config).to(device)

    # -- DataCollator
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # -- Inference
    data_loader = DataLoader(test_dataset, batch_size=training_args.eval_batch_size, collate_fn=data_collator)
    predictions = []
    for batch in tqdm(data_loader) :

        for key in batch :
            batch[key] = batch[key].to(device)

        preds = model.generate(batch["input_ids"], 
            max_length=training_args.generation_max_length, 
            num_beams=1, 
            eos_token_id=tokenizer.eos_token_id, 
            pad_token_id=tokenizer.pad_token_id
        )
        preds = preds.detach().cpu().numpy()
        predictions.extend(preds.tolist())

    # -- Postprocessing predictions
    pred_list = [preprocess(pred[1:], tokenizer) for pred in predictions]
    pred_strings = [tokenizer.decode(pred).split(", ") for pred in pred_list]
    tag_list = list(tag_dict.keys())

    pred_entities = []
    pred_labels = []

    pred_size = len(pred_strings)
    for i in range(0, pred_size, 6) :
        preds = pred_strings[i:i+6]

        entities = []
        labels = []

        for j in range(6) :
            pred = preds[j]
            if len(pred) == 1 and pred[0] == "없음" :
                continue
            else :
                entities.extend(preds[j])
                labels.extend([tag_list[j]] * len(preds[j]))

        pred_entities.append(entities)
        pred_labels.append(labels)

    # -- Saving results
    items_list = []
    pred_size = len(pred_entities)
    for i in range(pred_size) :
        items = []
       
        for pair in zip(pred_entities[i], pred_labels[i]) :
            w, t = pair
            item = "<" + w + ":" + t + ">"
            items.append(item)
        
        if len(items) == 0 :
            item_str = ""
        else :
            item_str = ", ".join(items)
        items_list.append(item_str)
            
    results = pd.DataFrame({"Named Entity" : items_list})
    path = os.path.join(training_args.output_dir, OUTPUT_FILE)
    results.to_csv(path, index=False)


if __name__ == "__main__":
    main()
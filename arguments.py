from typing import Optional
from dataclasses import dataclass, field
from transformers import Seq2SeqTrainingArguments

@dataclass
class ModelArguments:
    PLM: str = field(
        default="KETI-AIR/ke-t5-base",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    save_path: str = field(
        default="results", metadata={"help": "Path to save checkpoint from fine tune model"},
    )
    

@dataclass
class DataTrainingArguments:
    max_input_length: int = field(
        default=512, metadata={"help": "Max length of input sequence"},
    )
    max_output_length: int = field(
        default=128, metadata={"help": "Max length of output sequence"},
    )  
    data_dir: str = field(
        default="data", metadata={"help": "path of data directory"}
    )
    train_data_file: str = field(
        default="klue_ner_train_80.t", metadata={"help": "name of train data"}
    )
    eval_data_file: str = field(
        default="klue_ner_test_80.t", metadata={"help": "name of test data"}
    )

@dataclass
class TrainingArguments(Seq2SeqTrainingArguments):
    report_to: Optional[str] = field(default="wandb")
    output_dir: str = field(
        default="checkpoints",
        metadata={"help": "model output directory"}
    )
    fold_size: int = field(
        default=5, metadata={"help": "The number of fold for K-Fold validatin"},
    )
    use_noam: bool = field(
        default=False, metadata={"help" : "flag for using noam scheduelr"}
    )



@dataclass
class LoggingArguments:
    dotenv_path: Optional[str] = field(
        default="wandb.env", metadata={"help": "input your dotenv path"},
    )
    project_name: Optional[str] = field(
        default="KT-Dev", metadata={"help": "project name"},
    )
    group_name: Optional[str] = field(
        default="Named Entity Distinguish", metadata={"help": "group name"},
    )

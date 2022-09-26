import os
import json
import torch
import datasets
from tqdm import tqdm
from datasets import Dataset
from collections.abc import Mapping
from typing import Optional, List, Tuple, Union, Any, Dict
from transformers import Trainer, Seq2SeqTrainer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR


class Trainer(Trainer):
    def __init__(self, *args, eval_examples=None, postprocessor=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.postprocessor = postprocessor

    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
    ):
        eval_dataset = self.eval_dataset
        eval_examples = self.eval_examples
        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        if isinstance(eval_dataset, datasets.Dataset):
            eval_dataset.set_format(
                type=eval_dataset.format["type"], columns=list(eval_dataset.features.keys()),
            )

        output = self.prediction_loop(
            eval_dataloader,
            description="Evaluation",
            prediction_loss_only=None,
            ignore_keys=ignore_keys,
        )
        metrics = output.metrics

        checkpoint_dir = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        if self.postprocessor is not None:
            eval_sentences = eval_examples["sentences"]
            print("\nPostprocessing")
            pred_tag_words, pred_tag_names = self.postprocessor(output.predictions, eval_sentences)                

            print("\nScoring")    
            label_tag_words, label_tag_names = (eval_examples["tag_words"], eval_examples["tag_names"])
            score = self.get_score(pred_tag_words, 
                pred_tag_names,
                label_tag_words,
                label_tag_names
            )

            for key in score :
                metrics[key] = score[key]

            for key in list(metrics.keys()):
                if not key.startswith(f"{metric_key_prefix}_"):
                    metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

            self.log(metrics)

        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics
        )
        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return metrics


    def get_score(
        self, 
        pred_tag_words, 
        pred_tag_names, 
        label_tag_words,
        label_tag_names
    ) :
        eval_f1 = 0.0
        eval_precision = 0.0
        eval_recall = 0.0
        
        eval_size = len(pred_tag_words)
        for i in tqdm(range(eval_size)) :
            prediction = (pred_tag_words[i], pred_tag_names[i])
            label = (label_tag_words[i], label_tag_names[i])
            
            precision, recall, f1 = self.compute_score(prediction, label)
            eval_precision += precision
            eval_recall += recall
            eval_f1 += f1

        eval_f1 /= eval_size
        eval_precision /= eval_size
        eval_recall /= eval_size
        return {"f1" : eval_f1, "precision" : eval_precision, "recall" : eval_recall}


    def compute_score(self, prediction, label) :
        pred_tag_words, pred_tag_names = prediction
        label_tag_words, label_tag_names = label

        l_tag_num = len(label_tag_words)
        p_tag_num = len(pred_tag_words)
        p_tag_flag = [False] * p_tag_num 

        true_p = 0.0
        false_p = 0.0
        false_n = 0.0

        for i in range(l_tag_num) :
            l_tag_word = label_tag_words[i]
            l_tag_name = label_tag_names[i]

            flag = False

            for j in range(p_tag_num) :
                p_tag_word = pred_tag_words[j]
                p_tag_name = pred_tag_names[j]

                if (l_tag_word in p_tag_word) or (p_tag_word) in (l_tag_word) :
                    if l_tag_name == p_tag_name :
                        if p_tag_flag[j] == False :
                            true_p += 1
                            p_tag_flag[j] = True
                            flag = True

            if flag == False :
                false_n += 1.0

        for p_flag in p_tag_flag :
            if p_flag == False :
                false_p += 1.0

        if true_p == 0.0 :
            precision = 0.0
            recall = 0.0
            f1 = 0.0
        else :
            precision = true_p / (true_p + false_p)
            recall = true_p / l_tag_num
            f1 = 2/(1/precision + 1/recall)

        return precision, recall, f1



class Seq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _prepare_input(self, data: Union[torch.Tensor, Any]) -> Union[torch.Tensor, Any]:
        """
        Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
        """
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = dict(device=self.args.device)
            if self.deepspeed and data.dtype != torch.int64:
                # NLP models inputs are int64 and those get adjusted to the right dtype of the
                # embedding. Other models such as wav2vec2's inputs are already float and thus
                # may need special handling to match the dtypes of the model
                kwargs.update(dict(dtype=self.args.hf_deepspeed_config.dtype()))
            return data.to(**kwargs)
        return data

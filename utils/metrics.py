import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score
from transformers import EvalPrediction

class Seq2SeqMetrics :

    def __init__(self, tokenizer) :
        self.tokenizer = tokenizer

    def compute_metrics(self, pred: EvalPrediction):
        predictions = pred.predictions
        pred_list = [self.preprocess(pred[1:]) for pred in predictions]
        pred_strings = [self.tokenizer.decode(pred).split(", ") for pred in pred_list]

        references = pred.label_ids
        ref_list = [self.preprocess(ref) for ref in references]
        ref_strings = [self.tokenizer.decode(ref).split(", ") for ref in ref_list]

        wrong_size = 0.0
        eval_f1 = 0.0
        eval_size = len(ref_strings)        
        for i in tqdm(range(eval_size)) :
            pred_str = pred_strings[i]
            ref_str = ref_strings[i]

            if len(pred_str) != len(ref_str) :
                f1 = 0.0
                wrong_size += 1.0
            else :
                f1 = f1_score(ref_str, 
                    pred_str, 
                    labels=list(set(ref_str)),
                    average='macro'
                )

            eval_f1 += f1
        
        eval_f1 /= eval_size
        return {"f1" : eval_f1, "wrong_size" : wrong_size}

    def preprocess(self, array) :
        array = array.tolist()

        eos_token_id = self.tokenizer.eos_token_id

        index = len(array) - 1
        if eos_token_id in array :
            index = array.index(eos_token_id)

        valid = array[:index]
        return valid


class NERMetrics :

    def __init__(self, ) :
        pass

    def compute_metrics(self, pred: EvalPrediction):
        predictions = pred.predictions
        pred_ids = np.argmax(predictions, axis=-1)
        references = pred.label_ids

        eval_f1 = 0.0
        eval_size = len(references)
        for i in tqdm(range(eval_size)) :
            ref = references[i].tolist()
            pred = pred_ids[i].tolist()

            last_index = ref.index(-100) if -100 in ref else len(ref) - 1
            ref = ref[:last_index]
            pred = pred[:last_index]

            f1 = f1_score(ref, 
                pred, 
                labels=np.unique(ref),
                average='macro'
            )
            eval_f1 += f1

        eval_f1 /= eval_size
        return {"f1" : eval_f1}

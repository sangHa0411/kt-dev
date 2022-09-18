
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
from transformers import EvalPrediction

class Metrics :

    def __init__(self, tokenizer) :
        self.tokenizer = tokenizer

    def compute_metrics(self, pred: EvalPrediction):
        predictions = pred.predictions
        pred_list = [self.preprocess(pred[1:]) for pred in predictions]
        pred_strings = [self.tokenizer.decode(pred).split(", ") for pred in pred_list]

        references = pred.label_ids
        ref_list = [self.preprocess(ref) for ref in references]
        ref_strings = [self.tokenizer.decode(ref).split(", ") for ref in ref_list]

        eval_f1, eval_acc = 0.0, 0.0
        eval_size = len(ref_strings)        
        for i in tqdm(range(eval_size)) :
            pred_str = pred_strings[i]
            ref_str = ref_strings[i]

            if len(pred_str) != len(ref_str) :
                f1 = 0.0
                acc = 0.0
                print(i)
            else :
                f1 = f1_score(ref_str, 
                    pred_str, 
                    labels=list(set(ref_str)),
                    average='macro'
                )
                acc = accuracy_score(ref_str, pred_str)

            eval_f1 += f1
            eval_acc += acc
        
        eval_f1 /= eval_size
        eval_acc /= eval_size
        return {"f1" : eval_f1, "acc" : eval_acc}

    def preprocess(self, array) :
        array = array.tolist()

        eos_token_id = self.tokenizer.eos_token_id

        index = len(array) - 1
        if eos_token_id in array :
            index = array.index(eos_token_id)

        valid = array[:index]
        return valid
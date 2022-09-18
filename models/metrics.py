
from tqdm import tqdm
from sklearn.metrics import f1_score
from transformers import EvalPrediction

class Metrics :

    def __init__(self, tokenizer) :
        self.tokenizer = tokenizer

    def compute_metrics(self, pred: EvalPrediction):
        predictions = pred.predictions
        pred_list = [self.preprocess(pred[1:]) for pred in predictions]
        pred_strings = [self.tokenizer.decode(pred[:1]) for pred in pred_list]

        references = pred.label_ids
        ref_list = [self.preprocess(ref) for ref in references]
        ref_strings = [self.tokenizer.decode(ref[:1]) for ref in ref_list]

        macro_f1 = f1_score(ref_strings, 
            pred_strings, 
            labels=[
                "수량",
                "날짜",
                "사람",
                "장소",
                "시간",
                "기관"
            ],
            average='macro'
        )
        return {"f1" : macro_f1}

    def preprocess(self, array) :
        array = array.tolist()
        array = [a for a in array if a != 0]
        return array
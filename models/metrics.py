
from transformers import EvalPrediction

class Metrics :

    def __init__(self, tokenizer) :
        self.tokenizer = tokenizer

    def compute_metrics(self, pred: EvalPrediction):
        predictinos = pred.predictions
        references = pred.label_ids
        breakpoint()
        return 0
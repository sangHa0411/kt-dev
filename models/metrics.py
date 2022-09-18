
from tqdm import tqdm
from sklearn.metrics import f1_score
from transformers import EvalPrediction

class Metrics :

    def __init__(self, tokenizer) :
        self.tokenizer = tokenizer
        # self.label_list = [
        #     "수량",
        #     "날짜",
        #     "사람",
        #     "장소",
        #     "시간",
        #     "기관"
        # ]

    def compute_metrics(self, pred: EvalPrediction):
        predictions = pred.predictions
        pred_list = [self.preprocess(pred[1:]) for pred in predictions]
        pred_strings = [self.tokenizer.decode(pred).split(", ") for pred in pred_list]

        references = pred.label_ids
        ref_list = [self.preprocess(ref) for ref in references]
        ref_strings = [self.tokenizer.decode(ref).split(", ") for ref in ref_list]

        score = 0.0
        eval_size = len(ref_strings)        
        for i in tqdm(range(eval_size)) :
            pred_str = pred_strings[i]
            ref_str = ref_strings[i]

            if len(pred_str) != len(ref_str) :
                macro_f1 = 0.0
            else :
                macro_f1 = f1_score(ref_str, pred_str, average='micro')

            score += macro_f1
        
        score /= eval_size
        return {"f1" : score}

    def preprocess(self, array) :
        array = array.tolist()

        eos_token_id = self.tokenizer.eos_token_id

        index = len(array) - 1
        if eos_token_id in array :
            index = array.index(eos_token_id)

        valid = array[:index]
        return valid
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
from transformers import EvalPrediction

class ScoreCalculator :

    def __init__(self) :
        pass

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


class Seq2SeqClassifyMetrics :

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
                wrong_size += 1.0

                if len(pred_str) < len(ref_str) :
                    gap_size = len(ref_str) - len(pred_str)
                    pred_str = pred_str + ["X"] * gap_size
                else :
                    pred_str = pred_str[:len(ref_str)]

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



class Seq2SeqSearchMetrics :

    def __init__(self, tokenizer, score_calculator, eval_examples, tag_list) :
        self.tokenizer = tokenizer
        self.score_calculator = score_calculator
        self.eval_examples = eval_examples
        self.tag_list = tag_list

    def compute_metrics(self, pred: EvalPrediction):
        eval_entities, eval_labels = self.eval_examples["entities"], self.eval_examples["labels"]

        predictions = pred.predictions
        pred_list = [self.preprocess(pred[1:]) for pred in predictions]
        pred_strings = [self.tokenizer.decode(pred).split(", ") for pred in pred_list]

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
                    labels.extend([self.tag_list[j]] * len(preds[j]))

            pred_entities.append(entities)
            pred_labels.append(labels)
        
        score = self.score_calculator.get_score(pred_entities, pred_labels, eval_entities, eval_labels)
        return score

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
        return {"token_f1" : eval_f1}

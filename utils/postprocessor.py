import numpy as np
from konlpy.tag import Mecab
from tqdm import tqdm

class NERPostprocessor :

    def __init__(self, tokenizer, max_length, tag_dict) :
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tag_dict = tag_dict
        self.mecab = Mecab()

    def __call__(self, predictions, sentences) :
        pred_ids = np.argmax(predictions, axis=-1)

        tokenized = self.tokenizer(
            sentences,
            return_token_type_ids=False,
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.max_length
        )

        offsets = tokenized.pop("offset_mapping")

        words_list = []
        labels_list = []

        for i in tqdm(range(len(pred_ids))) :
            pred = pred_ids[i]
            offset = offsets[i]

            tokens = self.tokenizer.tokenize(sentences[i])
            tokens.append(self.tokenizer.eos_token)

            pred_valid = pred[:len(offset)]
            sentence = sentences[i]

            words = []
            labels = []

            j = 0
            while j < len(pred_valid) :
                p = pred_valid[j]
                if p > 0 :
                    k = j + 1
                    while k < len(pred_valid) and (pred_valid[k] > 0 or tokens[k] == '▁') :
                        k += 1
                    
                    sub_offset = offset[j:k]
                    start_p = sub_offset[0][0]
                    end_p = sub_offset[-1][1]

                    word = sentence[start_p:end_p].strip()
                    words.append(word)

                    label = self.tag_dict[p]
                    labels.append(label)

                    j = k                
                else :
                    j += 1
            
            words_list.append(words)
            labels_list.append(labels)
                    
        return words_list, labels_list


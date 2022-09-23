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

        words = []
        labels = []

        for i in tqdm(range(len(pred_ids))) :
            pred = pred_ids[i]
            offset = offsets[i]
            sentence = sentences[i]
            pos = self.get_pos(sentence)

            char_list = list(sentence)
            tag_list = []

            if offset[0][0] == offset[1][0] :
                offset = offset[1:]

            for j in range(len(offset)-1) :
                start_p, end_p = offset[j]
                tag = pred[j]
                tag_list.extend([tag] * (end_p - start_p))
            tag_list.append(0)

            prev = 0
            k = 1
            word_list = []
            label_list = []
            while k < len(tag_list) :
                if tag_list[k] != tag_list[k-1] :
                    if tag_list[k-1] > 0 :
                        l = k-1
                        while pos[l] == "J" :
                            l -= 1

                        word = [char_list[j] for j in range(prev, l+1)]
                        word = "".join(word).strip()
                        word_list.append(word)
                        label_list.append(self.tag_dict[tag_list[l]])
                    
                    prev = k
                k += 1
            
            words.append(word_list)
            labels.append(label_list)

        return words, labels

    def get_pos(self, data) :
        i = 0
        pos_label = []
        pos_data = self.mecab.pos(data)

        for pos in pos_data :
            word, tag = pos

            for j in range(len(word)) :
                pos_label.append(tag[0])
                i += 1
            
            if i < len(data) and data[i] == " " :
                pos_label.append("U")
                i += 1

        return pos_label
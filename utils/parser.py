import re
import pandas as pd
from tqdm import tqdm
from datasets import Dataset


class Seq2SeqParser :

    def __init__(self, ) :
        pass

    def extract(self, data) :

        tag_iter = re.finditer("<[^:]+:[A-Z]{2}>", data)

        entities = []
        labels = []

        for tag in tag_iter :
            group = tag.group()
            word, label = group[1:-1].split(':')

            data = re.sub(group, word, data)
            entities.append(word)
            labels.append(label)

        return data, entities, labels

    def __call__(self, dataset) :

        sentences = []
        entities = []
        labels = []

        for data in tqdm(dataset) :
            sen, entity, label = self.extract(data)

            sentences.append(sen)
            entities.append(entity)
            labels.append(label)

        df = pd.DataFrame(
            {
                "sentences" : sentences, 
                "entities" : entities,
                "labels" : labels
            }
        )
        dset = Dataset.from_pandas(df)
        return dset


class NERParser :

    def __init__(self, ) :
        pass

    def extract(self, dataset) :

        word_list = []
        tag_list = []

        for data in dataset :
            tag_info = re.findall("<[^:]+:[A-Z]{2}>", data)

            words = []
            tags = []
            for t in tag_info :
                word, tag = t[1:-1].split(":")
                words.append(word)
                tags.append(tag)

            word_list.append(words)
            tag_list.append(tags)

        return word_list, tag_list

    def preprocess(self, data) :

        label = ['O'] * len(data)
        tag_iter = re.finditer("<[^:]+:[A-Z]{2}>", data)

        for it in tag_iter :

            start_p, end_p = it.span()
            group = it.group()
            word, tag = group[1:-1].split(':')

            start_pos = start_p + 1
            label[start_pos] = tag
            for i in range(start_pos+1, start_pos+len(word)) :
                label[i] = tag

            label[start_p] = "X"
            for i in range(start_pos+len(word), end_p) :
                label[i] = "X"

        chars = [*data]

        x = []
        y = []
        for i, c in enumerate(chars) :
            l = label[i]
            if l != "X" :
                x.append(c)
                y.append(l)
        
        sen = ''.join(x)
        sen, y = self.delete_blank(sen, y)
        return sen, y

    def delete_blank(self, sentence, label) :

        blank_flag = re.search("\s{2,}", sentence)

        while blank_flag is not None :
            start_p, end_p = blank_flag.span()
            sentence = sentence[:start_p+1] + sentence[end_p:]
            label = label[:start_p+1] + label[end_p:]

            blank_flag = re.search("\s{2,}", sentence)

        return sentence, label


    def __call__(self, dataset) :

        sentences = []
        labels = []

        for data in tqdm(dataset) :
            sen, label = self.preprocess(data)

            sentences.append(sen)
            labels.append(label)

        df = pd.DataFrame(
            {
                "sentences" : sentences, 
                "labels" : labels
            }
        )
        dset = Dataset.from_pandas(df)
        return dset
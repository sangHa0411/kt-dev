import collections
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import Dataset

class Seq2SeqClassifyPreprocessor :
    def __init__(self, tokenizer, tag_dict) :
        self.tokenizer = tokenizer
        self.tag_dict = tag_dict

    def __call__(self, dataset) :

        inputs = []
        labels = []

        size = len(dataset["sentences"])
        for i in range(size) :
            sentence = dataset["sentences"][i]
            entities = dataset["entities"][i]
            label = dataset["labels"][i]

            input_sen = "개체명 : " + ", ".join(entities) + ". 문서 : " + sentence
            inputs.append(input_sen)
            label = [self.tag_dict[l] for l in label]
            sep_token = ", "
            label_sen = sep_token.join(label)
            labels.append(label_sen)


        dataset["inputs"] = inputs
        dataset["labels"] = labels
        return dataset


class Seq2SeqSearchPreprocessor :
    def __init__(self, tag_dict) :
        self.tag_dict = tag_dict

    def preprocess4test(self, dataset) :
        inputs = []
        size = len(dataset)
        for i in tqdm(range(size)) :
            sentence = dataset[i]["sentences"]

            for tag in self.tag_dict :
                tag_name = self.tag_dict[tag]
                prefix = "개체 유형 : " + tag_name

                input_sen = prefix + ", " + sentence
                inputs.append(input_sen)

        df = pd.DataFrame({"inputs" : inputs})
        dset = Dataset.from_pandas(df)
        return dset

    def preprocess4train(self, dataset, eval_flag) :

        inputs = []
        labels = []

        size = len(dataset)
        for i in tqdm(range(size)) :
            sentence = dataset[i]["sentences"]
            entities = dataset[i]["entities"]
            label = dataset[i]["labels"]

            default_dict = collections.defaultdict(list)
            for e, l in zip(entities, label) :
                default_dict[l].append(e)

            input_sens = []
            output_sens = []

            for tag in self.tag_dict :
                tag_name = self.tag_dict[tag]
                word_list = default_dict[tag]

                prefix = "개체 유형 : " + tag_name
                input_sen = prefix + ", " + sentence
                if len(word_list) == 0 :

                    if eval_flag :
                        output_sen = "없음"
                    else :
                        flag = np.random.randint(2)

                        if flag == 1 :
                            output_sen = "없음"
                        else :
                            continue
                else :
                    output_sen = ", ".join(word_list)

                input_sens.append(input_sen)
                output_sens.append(output_sen)

            inputs.extend(input_sens)
            labels.extend(output_sens)

        df = pd.DataFrame({"inputs" : inputs, "labels" : labels})
        dset = Dataset.from_pandas(df)
        return dset

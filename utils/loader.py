
import os
import random

class Loader :

    def __init__(self, data_dir, data_name) :
        self.data_dir = data_dir
        self.data_name = data_name

    def load(self, test_flag=False) :
        raw_data = os.path.join(self.data_dir, self.data_name)

        with open(raw_data, "r") as f :
            sentences = f.readlines()

        dataset = []
        for sen in sentences :
            dataset.append(sen[:-1])

        if not test_flag :
            random.shuffle(dataset)
        return dataset

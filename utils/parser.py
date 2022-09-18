import re
import pandas as pd
from tqdm import tqdm
from datasets import Dataset

def extract(data) :

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


def parsing(dataset) :

    sentences = []
    entities = []
    labels = []

    for data in tqdm(dataset) :
        sen, entity, label = extract(data)

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
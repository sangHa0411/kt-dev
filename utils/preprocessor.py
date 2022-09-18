

class Preprocessor :
    def __init__(self, tokenizer) :
        self.tokenizer = tokenizer
        self.label_dict = {
            "QT" : "수량",
            "DT" : "날짜",
            "PS" : "사람",
            "LC" : "장소",
            "TI" : "시간",
            "OG" : "기관"
        }

    def __call__(self, dataset) :

        inputs = []
        labels = []

        size = len(dataset['sentences'])
        for i in range(size) :
            sentence = dataset['sentences'][i]
            entities = dataset['entities'][i]

            if "labels" in dataset :
                label = dataset['labels'][i]
            else :
                label = None
            
            input_sen = "개체명 : " + ", ".join(entities) + ". 문서 : " + sentence
            inputs.append(input_sen)

            if label is not None :
                label = [self.label_dict[l] for l in label]
                sep_token = ", "
                label_sen = sep_token.join(label)
                labels.append(label_sen)
            else :
                labels.append("")

        dataset['inputs'] = inputs
        dataset['labels'] = labels
        return dataset
import re
import random
import collections

class Spliter :

    def __init__(self, dataset, fold_size) :
        self.dataset = dataset
        self.fold_size = fold_size
        self.mapping = self.build(dataset)

    def build(self, dataset) :
        tag_mapping = collections.defaultdict(list)
        for i, data in enumerate(dataset) :
            tag_list = re.findall("<[^:]+:[A-Z]{2}>", data)

            for tag in tag_list :
                _, tag = tag[1:-1].split(':')
                tag_mapping[tag].append(i)

        for tag in tag_mapping :
            id_list = tag_mapping[tag]
            random.shuffle(id_list)
            tag_mapping[tag] = id_list

        return tag_mapping

    def get_dataset(self, fold_num) :
        assert 0 <= fold_num and fold_num < self.fold_size

        train_ids = []
        validation_ids = []

        for tag in self.mapping :

            id_list = self.mapping[tag]
            total_size = len(id_list)
            block_size = int(total_size / self.fold_size)

            train_id_list = id_list[:fold_num*block_size] + id_list[(fold_num+1)*block_size:]
            validation_id_list = id_list[fold_num*block_size : (fold_num+1)*block_size]

            train_ids.extend(train_id_list)
            validation_ids.extend(validation_id_list)

        validation_ids = list(set(validation_ids) - set(train_ids))
        train_ids = list(set(train_ids))
        return train_ids, validation_ids
        

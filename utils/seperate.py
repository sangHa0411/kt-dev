import re
import random
import collections

class Spliter :

    def __init__(self, dataset, fold_size) :
        self.dataset = dataset
        self.fold_size = fold_size
        self.mapping = self.build(dataset)

    def build(self, dataset) :
        tag_mapping = {
            0 : collections.defaultdict(list),
            1 : collections.defaultdict(list),
            2 : collections.defaultdict(list)
        }
        
        for i, data in enumerate(dataset) :
            tag_list = re.findall("<[^:]+:[A-Z]{2}>", data)

            for tag in tag_list :
                word, tag_name = tag[1:-1].split(":")
    
                if len(tag_list) == 1 :
                    tag_mapping[0][tag_name].append(i)
                elif len(tag_list) < 4 :
                    tag_mapping[1][tag_name].append(i)
                else :
                    tag_mapping[2][tag_name].append(i)

        for tag_size in tag_mapping :
            mapping = tag_mapping[tag_size]
            for tag in mapping :
                id_list = mapping[tag]
                mapping[tag] = sorted(id_list)
            tag_mapping[tag_size] = mapping

        return tag_mapping

    def get_dataset(self, fold_num) :
        assert 0 <= fold_num and fold_num < self.fold_size

        validation_ids = []
        for tag_size in self.mapping :
            mapping  = self.mapping[tag_size]
            for tag in mapping :
                id_list = mapping[tag]

                total_size = len(id_list)
                block_size = int(total_size / self.fold_size)

                validation_id_list = id_list[fold_num*block_size : (fold_num+1)*block_size]
                validation_ids.extend(validation_id_list)

        validation_ids = list(set(validation_ids))
        train_ids = list(set(range(len(self.dataset))) - set(validation_ids))
        return train_ids, validation_ids
        

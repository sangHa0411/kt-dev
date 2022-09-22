
class Seq2SeqEncoder :
    def __init__(self, tokenizer, max_input_length, max_output_length) :
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

    def __call__(self, examples):
        inputs = examples['inputs']
        model_inputs = self.tokenizer(inputs, 
            max_length=self.max_input_length, 
            return_token_type_ids=False, 
            truncation=True
        )

        if "labels" in examples :
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(examples["labels"],
                    max_length=self.max_output_length, 
                    return_token_type_ids=False, 
                    truncation=True
                )

            model_inputs["labels"] = labels["input_ids"]
        return model_inputs


class NEREncoder :

    def __init__(self, tokenizer, max_length, label_dict=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_dict = label_dict

    def __call__(self, dataset):

        docs = dataset["sentences"]
        model_inputs = self.tokenizer(
            docs,
            return_token_type_ids=False,
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.max_length,
        )

        offset_mappings = model_inputs.pop("offset_mapping")

        if "labels" in dataset :
            raw_labels = dataset["labels"]

            model_labels = []
            for i, offset_list in enumerate(offset_mappings) :
                tokenized_labels = []
                raw_label = raw_labels[i]
                for offset in offset_list :
                    start_p, end_p = offset
                    if start_p == 0 and end_p == 0 :
                        label = "O"    
                    else :

                        label = "O"
                        for j in range(start_p, end_p) :
                            char_label = raw_label[j]

                            if char_label != "O" :
                                label = char_label
                                break
                        
                    tok_label = self.label_dict[label]
                    tokenized_labels.append(tok_label)
                model_labels.append(tokenized_labels)
            
            model_inputs["labels"] = model_labels
        
        return model_inputs
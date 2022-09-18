
class Encoder :
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
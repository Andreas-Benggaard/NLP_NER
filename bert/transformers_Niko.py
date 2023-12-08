# import os
# import torch
# os.environ['TRANSFORMERS_USE_TF'] = 'False'
# from transformers import AutoTokenizer, AutoModelForTokenClassification

# model = "xlm-roberta-large-finetuned-conll03-english"
# tokenizer = AutoTokenizer.from_pretrained(model)
# model = AutoModelForTokenClassification.from_pretrained(model)


# # from transformers import pipeline 
# # classifier = pipeline('ner',model=model,tokenizer=tokenizer)
# # s = classifier('Alya told Jasmine that Andrew could pay with cash')
# sequence = 'Alya told Jasmine that Andrew could pay with cash'
# print(tokenizer(sequence))
# tokens = tokenizer.tokenize(sequence)
# print(tokens)
# ids = tokenizer.convert_tokens_to_ids(tokens)
# print(ids)
# decoded_string = tokenizer.decode(ids)
# print(decoded_string)


# # print(s)





from datasets import load_dataset

dataset = load_dataset("yelp_review_full")


from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
print(tokenized_datasets)

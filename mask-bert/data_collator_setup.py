# data_collator_setup.py
from transformers import DataCollatorForWholeWordMask

def initialize_data_collator(tokenizer):
    return DataCollatorForWholeWordMask(
        tokenizer=tokenizer,
        return_tensors="pt",
    )

def demonstrate_data_collator(data_collator, tokenized_datasets, tokenizer, num_examples=2):
    samples = [tokenized_datasets["train"][i] for i in range(num_examples)]
    for sample in samples:
        _ = sample.pop("word_ids")

    for chunk in data_collator(samples)["input_ids"]:
        print(f"\n'>>> {tokenizer.decode(chunk)}'")

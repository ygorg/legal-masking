# data_collator_setup.py
import logging

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


    logging.info("====================================================================")
    logging.info("Example of masked document")
    logging.info("====================================================================")

    for chunk in data_collator(samples)["input_ids"]:
        logging.info(f"\n'>>> {tokenizer.decode(chunk)}'")

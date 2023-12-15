# data_setup.py
import os
import json
import logging

from datasets import Dataset

def load_dataset(data_dir, num_examples=None):

    if num_examples is not None:
        logging.warning("Loading only {num_examples} examples, this is not expected for training/testing.")

    # ... [Existing function code remains unchanged]
    def load_dataset_from_json(file_path):
        try:
            with open(file_path, 'r') as file:
                data_list = [json.loads(line) for line in file]
            data = {key: [dic[key] for dic in data_list] for key in data_list[0]}
            return Dataset.from_dict(data)
        except Exception as e:
            logging.error(f"Failed to load or process data from {file_path}. Error: {e}")
            return None

    dataset_dict = {}
    for split in ['train', 'validation', 'test']:
        file_path = os.path.join(data_dir, f'{split}.json')
        if not os.path.exists(file_path):
            logging.error(f"File not found: {file_path}")
            exit(0)

        logging.info(f"Loading dataset from {file_path}")

        dataset = load_dataset_from_json(file_path)
        if not dataset:
            logging.error(f"No data loaded for {split}.")
        else:
            logging.info(f"{len(dataset)} examples loaded.")
        
        if num_examples is not None:
            dataset_dict[split] = dataset.select(range(min(num_examples, len(dataset))))
        else:
            dataset_dict[split] = dataset

    if not dataset_dict:
        exit(0)
            

    return dataset_dict


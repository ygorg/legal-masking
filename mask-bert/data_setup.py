import os
import json
import logging

from datasets import Dataset


def load_dataset(data_dir, num_examples=None):
    if num_examples is not None:
        logging.warning(f"Loading only {num_examples} examples, this is not expected for training/testing.")

    # ... [Existing function code remains unchanged]
    def load_dataset_from_json(file_path):
        with open(file_path, 'r') as file:
            if num_examples is None:
                data_list = [json.loads(line) for line in file]
            else:
                data_list = []
                for i in range(num_examples):
                    line = next(file, None)
                    if line is None:
                        break
                    data_list.append(json.loads(line))

        data = {key: [dic[key] for dic in data_list] for key in data_list[0]}
        return Dataset.from_dict(data)

    dataset_dict = {}
    for split in ['train', 'validation', 'test']:
        file_path = os.path.join(data_dir, f'{split}.json')

        logging.info(f"Loading dataset from {file_path}")

        dataset = load_dataset_from_json(file_path)
        if not dataset:
            logging.error(f"No data loaded for {split}.")
        else:
            logging.info(f"{len(dataset)} examples loaded.")

        dataset_dict[split] = dataset

    return dataset_dict

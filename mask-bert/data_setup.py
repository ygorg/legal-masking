import os
import json
from datasets import Dataset

def create_small_datasets(data_dir, num_examples=100):
    def load_dataset_from_json(file_path):
        try:
            with open(file_path, 'r') as file:
                data_list = [json.loads(line) for line in file]
            data = {key: [dic[key] for dic in data_list] for key in data_list[0]}
            return Dataset.from_dict(data)
        except Exception as e:
            print(f"Failed to load or process data from {file_path}. Error: {e}")
            return None

    dataset_dict = {}
    for split in ['train', 'validation', 'test']:
        file_path = os.path.join(data_dir, f'{split}.json')
        if os.path.exists(file_path):
            print(f"Loading dataset from {file_path}")
            dataset = load_dataset_from_json(file_path)
            if dataset:
                dataset_dict[split] = dataset.select(range(min(num_examples, len(dataset))))
            else:
                print(f"No data loaded for {split}.")
        else:
            print(f"File not found: {file_path}")

    return dataset_dict

# Usage
#data_directory = './data'  # Make sure to use the correct path
#small_datasets = create_small_datasets(data_directory, 100)

# Check if data is returned
#if small_datasets:
#    for split, ds in small_datasets.items():
#        print(f"{split} dataset: {len(ds)} records")
#else:
#    print("No datasets were created.")

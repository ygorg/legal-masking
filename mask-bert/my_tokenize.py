from transformers import AutoModelForMaskedLM, AutoTokenizer
from data_setup import create_small_datasets

model_checkpoint = "models/bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
chunk_size = 128

# Tokenization Function
def tokenize_function(examples):
    result = tokenizer(examples["text"])
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result

# Group Texts Function
def group_texts(examples):
    # Process each example (document) individually
    result = {key: [] for key in examples.keys()}
    for index in range(len(examples["input_ids"])):
        # Extract single document
        single_doc = {key: examples[key][index] for key in examples.keys()}
        # Compute length
        doc_length = len(single_doc["input_ids"])
        # Adjust length to fit the chunk size
        doc_length = (doc_length // chunk_size) * chunk_size
        # Segment into chunks
        for i in range(0, doc_length, chunk_size):
            chunk = {key: single_doc[key][i:i + chunk_size] for key in single_doc.keys()}
            # Add chunk to result
            for key in chunk.keys():
                result[key].append(chunk[key])
    # Duplicate input_ids to labels if needed
    result["labels"] = result["input_ids"].copy()
    return result


# Load and prepare datasets
data_dir = './data'  # Specify your data directory
small_datasets = create_small_datasets(data_dir, 100)

print(small_datasets)

# Apply tokenization to each dataset in the dictionary
tokenized_datasets = {split: dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=['id', 'text', 'sector', 'descriptor', 'year', '__index_level_0__']
    ) for split, dataset in small_datasets.items()}

# Apply the 'group_texts' function to each dataset in the dictionary
for split in tokenized_datasets.keys():
    tokenized_datasets[split] = tokenized_datasets[split].map(group_texts, batched=True)

# Output example
print(tokenizer.decode(tokenized_datasets["train"][0]["input_ids"]))

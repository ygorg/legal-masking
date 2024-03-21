import os
import json
import zipfile
import pandas as pd

# Download version 2 of scotus-opinion corpus
# https://www.kaggle.com/datasets/gqfiddler/scotus-opinions

DATA_PATH = os.path.join(os.path.dirname(__file__), 'corpus')
os.makedirs(DATA_PATH, exist_ok=True)


scotus_path = os.path.join(DATA_PATH, 'archive.zip')

if not os.path.exists(scotus_path):
    print('Please download "scotus" corpus from kaggle at ')
    print('https://www.kaggle.com/datasets/gqfiddler/scotus-opinions')
    print('and put it at `legal-masking/continuous-pretraining/corpus/archive.zip`.')
    exit()

print('Splitting SCOTUS to SCOTUS.zip...', end=' ', flush=True)
with zipfile.ZipFile(scotus_path) as zipf:
    with zipf.open("opinions_since_1970.csv") as f:
        df = pd.read_csv(f)

# add unique id to each document
df['id'] = range(1, len(df) + 1)
# set index to use `df.loc`
df = df.set_index('id')

# Split the DataFrame into train, validation, and test sets (adjust the proportions as needed)
with open(os.path.join(DATA_PATH, 'SCOTUS_split.json')) as f:
    scotus_split = json.load(f)

train_df = df.loc[scotus_split['train']].reset_index()
test_df = df.loc[scotus_split['test']].reset_index()
validation_df = df.loc[scotus_split['validation']].reset_index()

# Making sure there is no overlap in the splits
assert(set(train_df['id']) & set(test_df['id']) == set())
assert(set(train_df['id']) & set(validation_df['id']) == set())
assert(set(test_df['id']) & set(validation_df['id']) == set())
# Making sure all documents were kept
assert(len(test_df) + len(train_df) + len(validation_df) == len(df))

# Creating SCOTUS.zip
with zipfile.ZipFile(os.path.join(DATA_PATH, 'SCOTUS.zip'), 'w', zipfile.ZIP_DEFLATED) as zipf:
    for fn, df in [('train', train_df), ('validation', validation_df), ('test', test_df)]:
        with zipf.open(fn + '.jsonl', 'w') as file:
            df.to_json(file, lines=True, orient='records', index=False)
print('DONE')


##########
## Downloading sub corpuses from `lexlms/lex_files`
##########

ROOT_URL = "https://huggingface.co/datasets/lexlms/lex_files/resolve/main/"
# Assuming file paths are known
to_download = ['ecthr_cases.zip', 'eurlex.zip', 'indian_courts_cases.zip']

print('Downloading corpus from `lexlms/lex_files`')
for fn in to_download:
    f_path = os.path.join(DATA_PATH, fn)
    if os.path.exists(f_path):
        print(f"Using cached {fn}")
        continue
    url = ROOT_URL + fn
    os.system(f"wget -c --read-timeout=5 --tries=0 {url} -O {f_path}")


data_types = ['train', 'validation', 'test']
# Initialize empty DataFrames
combined_data = {data_type: pd.DataFrame() for data_type in data_types}

# Process each zip file
zip_files = ['ecthr_cases.zip', 'SCOTUS.zip', 'eurlex.zip', 'indian_courts_cases.zip']
for fn in zip_files:
    print(f'Loading {fn}...', end=' ', flush=True)
    f_path = os.path.join(DATA_PATH, fn)
    with zipfile.ZipFile(f_path) as zipf:
        # Load each data type
        for data_type in data_types:
            with zipf.open(f"{data_type}.jsonl") as f:
                df = pd.read_json(f, lines=True)
                # Only keep id and text
                df = df[['id', 'text']]
                # df['corpus'] = fn.split('.')[0]
                combined_data[data_type] = pd.concat([combined_data[data_type], df])
        print('DONE')

print('Saving combination of datasets...', end=' ', flush=True)
# Convert to Hugging Face Dataset
for split, df in combined_data.items():
    df.to_json(os.path.join(DATA_PATH, split + '.json'), lines=True, orient='records', index=False)
print('DONE')

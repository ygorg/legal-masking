# legal-masking

Instead of randomly masking words for the MLM (maked language modeling) objective in BERT-like models, we mask "important" words for the domain (terms), document (keywords), genre (meta-discourse).

```python
# Download all datasets in a cache then put them in the remote location
# BASH : export HF_DATASETS_CACHE=/home/gallina/datasets_cache/

for c in ['case_hold', 'ledgar', 'eurlex', 'ecthr_b', 'ecthr_a', 'scotus', 'unfair_tos']:
	datasets.load_dataset('lex_glue', c)

for c in ['canadian_crimes', 'canadian_sections', 'cjeu_terms', 'ecthr_terms', 'ecthr_articles', 'us_crimes', 'us_terms', 'contract_types', 'contract_sections']:
	datasets.load_dataset('lexlms/legal_lama', c)
```
## Dataset Overview



## Continuous fine-tuning:
```bash
cd mask-bert
python3 main.py --data-path /home/belfathi/mask-bert/data --mask-strategy default --num-epochs 4
```

## lexlama:
```bash
git clone https://github.com/coastalcph/lexlms
cd lexlms
model=PATH/TO/MODEL
python -m legal_lama.run_experiments --model_name_or_path ${model} --vocab_constraint true
```

```python
# Scores to csv
import re
# It needs to be the 'ecthr_articles' output because it is the first it holds everything
path_to_output = 'output/results_fair_eval/ecthr_articles/.._mask-bert_models_bert-base-uncased-e4-b16-tfidf-DEBUG1000_checkpoint-7500_constrained'
with open(path_to_output + '/info.log') as f:
    lines = [l.strip() for l in f if 'args' in l or 'global' in l]
out = ""
for l in lines:
    m = re.search(r"dataset_filename='data/(.*).jsonl'", l)
    if m:
        out += m.group(1) + '\n'
    else:
        a, b = l.split(':')
        out += f'{a}: {float(b)*100:2.2f}\n'
print(re.subn(r'(\w+)\nglobal MRR: ([\d\.]+)\nglobal Precision at 1: ([\d\.]+)', r'\1;\2;\3', out)[0])
```

## lex_glue
```bash
# git clone https://github.com/coastalcph/lex-glue
# I did some changed (scripts take CLI arguments)
git clone --branch update_scripts https://github.com/ygorg/lex-glue
cd lex-glue
model=../mask-bert/models/bert-base-uncased-e4-b16-default-DEBUG1000/checkpoint-7500
gpu=0  # same usage as CUDA_VISIBLE_DEVICES
for EXP_SCRIPT in scripts/run*.sh ; do
	PYTHONPATH="." $EXP_SCRIPT ${model} ${gpu}
```

# Language Model Adaptation to Specialized Domains through Selective Masking based on Genre and Topical Characteristics 

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

### Trainning Corpus 
This project utilizes a subset of the LeXFiles, a groundbreaking English multinational legal corpus. The LeXFiles encompasses a comprehensive collection of legal texts, featuring approximately 19 billion tokens across 11 distinct sub-corpora. It covers legislation and case law from six major English-speaking legal systems: the European Union (EU), the Council of Europe (CoE), Canada, the United States (US), the United Kingdom (UK), and India.
Objectives

Our aim was to create a balanced representation of these legal systems while managing computational resources effectively. To this end, our training corpus, derived from the LeXFiles, includes carefully selected sub-corpora, emphasizing four legal systems:

* European Union: EU Court Decisions
* Council of Europe: ECtHR Decisions
* India: Indian Court Decisions
* United States: A representative corpus extracted from Kaggle, courtesy of Garrett Fiddler

The final training corpus is approximately 4 GB, optimized for legal linguistic research and AI applications in the legal domain.


### Evaluation Strategy
**Masking Strategy Evaluation**

Our evaluation employs a probing benchmark with 8 sub-tasks, designed to assess the depth of legal knowledge in Pretrained Language Models (PLMs). This benchmark spans across all legal systems included in our training, providing a comprehensive evaluation of the PLMs.

**Legal Task Evaluation**

We utilize the LexGLUE benchmark, a robust assessment based on seven existing legal NLP datasets. LexGLUE is modeled after the criteria used in SuperGLUE and focuses on European and US legal systems. To extend our evaluation to the Indian legal system, we incorporated the LegalEval tasks, enriching our benchmark's scope.

### Installation and Setup

[Provide detailed installation and setup instructions here, including any dependencies or requirements necessary to work with the corpus.]

### Usage

[Explain how users can utilize the corpus in their research or AI applications, possibly with code snippets or example usage scenarios.]

### Credits

[If applicable, acknowledge any collaborators, contributors, or sources of third-party assets.]
### License

[Specify the licensing details for your project. Choose an appropriate license at choosealicense.com if necessary.]


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

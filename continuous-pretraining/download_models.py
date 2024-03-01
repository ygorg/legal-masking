import os
from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer

for model in ['bert-base-uncased',
              'nlpaueb/legal-bert-base-uncased']:
    AutoModelForMaskedLM.from_pretrained(model)\
        .save_pretrained(os.path.join('models', model))
    AutoTokenizer.from_pretrained(model)\
        .save_pretrained(os.path.join('models', model))

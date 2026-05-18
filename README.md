# Is Selective Masking A Key to Improving Domain Adaptation for Masked Language Model?

<img src="_assets/logo-selective-masking.jpg" align="left" alt="Project Logo" width="250"/>

<p>

<a href="https://doi.org/10.1145/3769126.3769216"><img src="https://img.shields.io/badge/Paper-ICAIL%202025-blue?style=flat-square&logo=read-the-docs" alt="Paper"></a>
<a href="https://creativecommons.org/licenses/by/4.0/"><img src="https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey?style=flat-square" alt="License"></a>
<img src="https://img.shields.io/badge/Conference-ICAIL%202025-orange?style=flat-square" alt="Conference">
<img src="https://img.shields.io/badge/Model-BERT-yellow?style=flat-square" alt="Model">

</p>

<br clear="left"/>

<p align="center">
  <b>Anas Belfathi &nbsp;·&nbsp; Ygor Gallina &nbsp;·&nbsp; Nicolas Hernandez &nbsp;·&nbsp; Laura Monceaux &nbsp;·&nbsp; Richard Dufour</b><br>
  <i>Nantes University, LS2N Lab, Nantes, France</i>
</p>

---

## Abstract

Recent advances in pre-trained language modeling have facilitated significant progress across various NLP tasks. Word masking during model training constitutes a pivotal component of language modeling in architectures like BERT. However, the prevalent method of word masking relies on random selection, potentially disregarding domain-specific linguistic attributes.

In this paper, we introduce an innovative masking approach leveraging **genre** and **topicality** information to tailor language models to specialized domains. Our method incorporates a ranking process that prioritizes words based on their significance, subsequently guiding the masking procedure. Experiments conducted using continual pre-training within the legal domain have underscored the efficacy of our approach on the LegalGLUE benchmark.

**TL;DR**: Instead of randomly masking words for the MLM (masked language modeling) objective in BERT-like models, we mask "important" words for the domain (terms), document (keywords) or genre (meta-discourse).

**Key findings:**
- Continual pre-training with selective masking offers **no significant advantage** over random masking — random masking remains a robust baseline for adapting pre-trained models.
- Training MLMs **from scratch** with topic-based or genre-based selective masking is effective, with topic-based masking yielding the biggest improvements (outperforming BERT-Random on 5 out of 6 tasks).

---

## Selective Masking Method

Our method prioritizes words using two domain-specific weighting strategies:

### Topic weighting (Tf-Idf)
Identifies content-bearing words that distinguish documents within the domain (e.g., "contract," "asylum," "tax"). Computed as:

```
s_topic(t, d) = tf_{d,t} · log((1 + |D|) / (1 + df_t) + 1)
```

### Genre weighting
Captures structural and stylistic markers that organize legal discourse (e.g., "ordered," "allowed," "acquitted"). These words appear consistently across documents with uniform frequency. Computed as:

```
s_genre(t) = (df_t / tf_t) × (1 - std(dtf_t) / max(dtf_t)) × (df_t / D)
```

### Word selection algorithm
After scoring, words are **randomly sampled proportionally to their weights** until 15% of tokens are masked — ensuring exposure to both high- and low-frequency domain words during training.

---

## Training Corpus

Our aim was to cover all legal systems appearing in the LexGLUE benchmark as well as [LegalEval](https://sites.google.com/view/legaleval/home), limited to a total size of **4GB** (~604M tokens across 181.8K documents).

| Sub-Corpus | # Documents | # Tokens |
|---|---|---|
| EU Case Law | 29.8K | 178.5M (29%) |
| ECtHR Case Law | 12.5K | 78.5M (13%) |
| Indian Case Law | 34.8K | 111.6M (19%) |
| SCOTUS Opinions | 104.7K | 235.5M (39%) |
| **Total** | **181.8K** | **604.1M** |

We use:
- A subset of [lexlms/lex_files](https://huggingface.co/datasets/lexlms/lex_files) covering EU, Council of Europe, UK, Canada, US, and India legal systems.
- The [SCOTUS Opinions](https://www.kaggle.com/datasets/gqfiddler/scotus-opinions) corpus from Kaggle.

---

## Results

Performance (Macro-F1) on LegalEval and LexGLUE benchmarks. † denotes significance over BERT-Random at the 0.01 level.

|  | Setting | RR | ECtHR (A) | ECtHR (B) | SCOTUS | EUR-LEX | LEDGAR | Avg. |
|---|---|---|---|---|---|---|---|---|
| SpanBERT | SOTA | 47.39 | 51.76 | 59.90 | 54.58 | 54.43 | 81.78 | 58.30 |
| ERNIE | SOTA | 46.31 | 45.26 | 52.29 | 42.35 | 52.64 | 80.54 | 53.23 |
| BERT-Random | Continual | 49.32 | 55.65 | 61.93 | 59.13 | 56.56 | **82.31** | 60.81 |
| BERT-Genre | Continual | 48.62 | 56.15 | 63.14 | 58.79 | 55.92 | 82.29 | 60.82 |
| **BERT-Topic** | **Continual** | **49.41** | **56.17†** | **64.14†** | **60.36** | **56.91** | 82.24 | **61.54** |
| BERT-Random | Scratch | 44.74 | 43.33 | 52.46 | 42.32 | 51.99 | 78.64 | 52.24 |
| BERT-Genre | Scratch | **46.83†** | 41.70 | 50.03 | 44.18 | 48.29 | **78.68** | 51.62 |
| **BERT-Topic** | **Scratch** | 44.66 | **45.26** | **57.08†** | **51.74†** | **52.74** | 78.64 | **55.02** |

---

## Continuous Pre-training

### 1. Download models

```bash
cd continuous-pretraining
python3 "download_models.py"
```

### 2. Recreate dataset

```bash
# Download SCOTUS opinions from Kaggle first:
# https://www.kaggle.com/datasets/gqfiddler/scotus-opinions
python3 "download_data.py"
```

### 3. Preprocess the corpus

```bash
python3 "preprocessing_dataset.py" \
    --data-path "corpus" \
    --model-checkpoint "models/bert-base-uncased" \
    --mask-strategy tfidf --chunk-size 512 \
    --cache-dir "cache_dir" --num-workers 8
```

### 4. Run continual pre-training

```bash
# Using a script (from continuous-pretraining/)
python3 "run_training.py" \
    --data-path "cache_dir" \
    --model-checkpoint "models/bert-base-uncased" \
    --mask-strategy tfidf --chunk-size 512 \
    --batch-size 16 --num-epochs 10 \
    --mask-choice weighted_random

# On JeanZay using SLURM
sbatch slurms/run_training_bert-CFT.sh
```

---

## Evaluation

We use the [LexGLUE](https://github.com/coastalcph/lex-glue) benchmark (7 legal NLP datasets covering European and US legal systems), extended with [LegalEval](https://sites.google.com/view/legaleval/home) tasks for the Indian legal system.

**Evaluation tasks:**

| Task | Description |
|---|---|
| ECtHR (A & B) | Identify violated / allegedly violated ECHR articles |
| SCOTUS | Categorize U.S. Supreme Court opinions (14 classes) |
| EUR-LEX | Assign EUROVOC concepts to EU legislation |
| LEDGAR | Classify US contract paragraphs (100 topics) |
| RR (LegalEval) | Rhetorical role labeling of Indian legal judgments |

### 1. Download LexGLUE and its data

```bash
cd evaluation
sh runme.sh
```

### 2. Run evaluation

```bash
# On JeanZay using SLURM
cd lex-glue
sbatch ../slurms/run_lexglue_BERT-TFIDF-RW.sh

# Using a script
TASK="eurlex"
MODEL_NAME="../../continuous-pretraining/models/bert-base-uncased-jz2-2-4-e10-b16-c512-tfidf-weighted_random-exall/checkpoint-3340"
MODEL_BASE_NAME="BERT-TFIDF-RW"
python experiments/${TASK}.py \
    --model_name_or_path ${MODEL_NAME} \
    --task ${TASK} --do_lower_case 'True' \
    --output_dir logs/${TASK}/${MODEL_BASE_NAME}/seed_1 \
    --do_train --do_eval --do_pred \
    --num_train_epochs 20 \
    --learning_rate 3e-5 \
    --per_device_train_batch_size 16 \
    --fp16
```

### 3. Display results

```bash
python3 show_results.py --logdir lex-glue/logs
```

---

## Citation

If you use this code or find our work useful, please cite:

```bibtex
@inproceedings{10.1145/3769126.3769216,
  author    = {Belfathi, Anas and Gallina, Ygor and Hernandez, Nicolas and Monceaux, Laura and Dufour, Richard},
  title     = {Is Selective Masking A Key to Improving Domain Adaptation for Masked Language Model?},
  booktitle = {Proceedings of the Twentieth International Conference on Artificial Intelligence and Law},
  series    = {ICAIL '25},
  year      = {2025},
  pages     = {409--413},
  publisher = {Association for Computing Machinery},
  address   = {New York, NY, USA},
  doi       = {10.1145/3769126.3769216},
  url       = {https://doi.org/10.1145/3769126.3769216}
}
```

---

## Acknowledgments

This work was granted access to the HPC resources of **IDRIS** under the allocations 2023-AD011014882 and 2023-AD011014767, provided by **GENCI**.

This research was funded in whole or in part by **l'Agence Nationale de la Recherche (ANR)**, project ANR-22-CE38-0004.
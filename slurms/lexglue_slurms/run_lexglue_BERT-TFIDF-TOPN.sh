#!/bin/bash
#SBATCH --job-name=b_tf_tn
#SBATCH --output=./job_out_err/run_evaluation_BERT_TFIDF_TOPN_%A_%a.out
#SBATCH --error=./job_out_err/run_evaluation_BERT_TFIDF_TOPN_%A_%a.err
#SBATCH --constraint=v100-32g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=08:00:00
#SBATCH --hint=nomultithread
#SBATCH --array=1-18%18

module purge
module load pytorch-gpu/py3/2.1.1

set -x
MODEL_NAME="/gpfsscratch/rech/zsl/upt42hk/MetaBERT_ygor/legal-masking/models/bert-base-uncased-jz4-4-4-e10-b16-c512-tfidf-top_n-exall/checkpoint-4453"
MODEL_BASE_NAME="bert-tfidf-topn"
CACHE_DIR="./data"
LOWER_CASE='True'
BATCH_SIZE=16
ACCUMULATION_STEPS=1
REPORT_TO='none'

# Calculer SEED et TASK index basé sur SLURM_ARRAY_TASK_ID
TASK_INDEX=$(( (SLURM_ARRAY_TASK_ID - 1) / 3 ))
SEED=$(( (SLURM_ARRAY_TASK_ID - 1) % 3 + 1 ))

TASKS=(
    'ecthr_a'
    'ecthr_b'
    'unfair_tos'
    'eurlex'
    'ledgar'
    'scotus'
)

TASK=${TASKS[$TASK_INDEX]}

COMMAND="srun -l python experiments/${TASK}.py \
    --model_name_or_path ${MODEL_NAME} \
    --do_lower_case ${LOWER_CASE} \
    --task ${TASK} \
    --output_dir logs/${TASK}/${MODEL_BASE_NAME}/seed_${SEED} \
    --cache_dir ${CACHE_DIR} \
    --do_train \
    --do_eval \
    --do_pred \
    --report_to ${REPORT_TO} \
    --overwrite_output_dir \
    --load_best_model_at_end \
    --metric_for_best_model micro-f1 \
    --greater_is_better True \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 5 \
    --num_train_epochs 20 \
    --learning_rate 3e-5 \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --seed ${SEED} \
    --fp16 \
    --fp16_full_eval \
    --gradient_accumulation_steps ${ACCUMULATION_STEPS} \
    --eval_accumulation_steps ${ACCUMULATION_STEPS}"

# Ajouter l'argument --hierarchical uniquement pour certaines tâches
if [[ "$TASK" == "ecthr_a" ]] || [[ "$TASK" == "ecthr_b" ]] || [[ "$TASK" == "scotus" ]]; then
    COMMAND+=" --hierarchical False"
fi

COMMAND+=" && python statistics.py"

# Exécuter la commande
eval $COMMAND

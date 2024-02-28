#!/bin/bash
#SBATCH --job-name=Lexglue-benchmark              # nom du job
#SBATCH --output=./job_out_err/run_evaluation_test_%A_%a.out # nom du fichier de sortie
#SBATCH --error=./job_out_err/run_evaluation_test_%A_%a.err  # nom du fichier d'erreur (ici en commun avec la sortie)
#SBATCH --constraint=v100-32g #partition 4GPU V100-32Go
#SBATCH --nodes=1                          # Utilisation de 1 nœud
#SBATCH --ntasks=1                         # 1 tâche par nœud
#SBATCH --gres=gpu:1                       # 1 GPU par nœud
#SBATCH --cpus-per-task=10                 # On réserve 10 cores CPU par tâche
#SBATCH --time=02:00:00
#SBATCH --hint=nomultithread
#SBATCH --account=zsl@v100
#SBATCH --array=1-3%1                      # Lance les tâches pour les seeds 1, 2, et 3, une à la fois

module purge
module load pytorch-gpu/py3/2.1.1

# Activation de l'écho des commandes
set -x

# Variables communes
MODEL_NAME="../../continued_pretraining/models/bert-base-uncased"
CACHE_DIR="./data"
LOWER_CASE='True'
BATCH_SIZE=16
ACCUMULATION_STEPS=1
TASK='unfair_tos'
HIERAR='False'
REPORT_TO='none'

# Récupération de la valeur de seed basée sur SLURM_ARRAY_TASK_ID
SEED=$SLURM_ARRAY_TASK_ID

# Correction de la syntaxe pour l'assignation des variables
# Et correction de la boucle pour générer les commandes

# Exécution du script Python avec les paramètres dynamiques pour chaque valeur de seed
srun -l python experiments/unfair_tos.py \
    --model_name_or_path ${MODEL_NAME} \
    --do_lower_case ${LOWER_CASE} \
    --task ${TASK} \
    --output_dir logs/${TASK}/${MODEL_NAME}/seed_${SEED} \
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
    --eval_accumulation_steps ${ACCUMULATION_STEPS} \
    && python statistics.py

#!/bin/bash
#SBATCH --job-name=be-base              # nom du job
#SBATCH --output=./job_out_err/run_training_baseline_%j.out          # nom du fichier de sortie
#SBATCH --error=./job_out_err/run_training_baseline_%j.err           # nom du fichier d'erreur (ici en commun avec la sortie)
#SBATCH --constraint=v100-32g #partition 4GPU V100-32Go
#SBATCH --nodes=4                          # Utilisation de 2 nœuds
#SBATCH --ntasks=4                         # 1 tâche par nœud
#SBATCH --gres=gpu:4                       # 1 GPU par nœud, donc 2 GPU au total
#SBATCH --cpus-per-task=40                 # On réserve 10 cores CPU par tâche (ajuster selon les besoins de votre application)
#SBATCH --time=20:00:00
#SBATCH --hint=nomultithread

set -e  # Stop script when an error occurs

module purge
module load pytorch-gpu/py3/2.1.1
 
# echo des commandes lancees
set -x
 
# execution
srun -l python3 \
     "run_training.py" \
     --data-path "cache_dir" \
     --model-checkpoint "models/bert-base-uncased" \
     --mask-strategy default --chunk-size 512  \
     --batch-size 16 --num-epochs 10 \
     --jean-zay-config 4-4-4

#!/bin/bash

#SBATCH --mail-user=adwi9965@colorado.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=32gb
#SBATCH --time=24:00:00
#SBATCH --qos=blanca-kann
#SBATCH --gres=gpu:2
#SBATCH --output=logs/optimal_tokenization_fintuneing.%j.log

# HEADERS ARE FOR CURC (which uses SLURM).

source /curc/sw/anaconda3/latest
conda activate optimal-tokenization

# TODO: add cli args for e.g. hyperparam grid search
python finetune_wmt.py \
       --post_tokenizer default \
       --model_name google/mt5-small

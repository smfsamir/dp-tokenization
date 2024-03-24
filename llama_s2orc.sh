#!/bin/bash
#SBATCH --job-name=scrape_lgbt_wiki
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND
#SBATCH --mail-user=smfsamir@uw.edu
#SBATCH --account=argon
#SBATCH --partition=gpu-l40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --mem=256G
#SBATCH --gpus=1
#SBATCH --time=4:00:00
#SBATCH --export=all
#SBATCH --output=/gscratch/argon/smfsamir/s2orc_llama_basic.out
#SBATCH --error=/gscratch/argon/smfsamir/s2orc_llama_basic.error
# Modules to use (optional).
source /gscratch/argon/smfsamir/uw_samir_env/bin/activate
# Your programs to run.
cd dp-tokenization
python main_analyze_s2orc.py
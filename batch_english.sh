#!/bin/bash
#SBATCH --job-name=ds606_project
#SBATCH --account=irohs_proj2
#SBATCH --partition=cn4_mangala
#SBATCH --qos=mangala
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source /users/student/prjstu/sravani.gunnu/miniconda3/etc/profile.d/conda.sh

conda activate venv

# export PYTHONPATH="/users/student/prjstu/sravani.gunnu/DS606_project/src:$PYTHONPATH"



python scripts/evaluate_per_language.py --input initial_malicious_final.csv --language english --batch-size 16


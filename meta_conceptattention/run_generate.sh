#!/bin/bash
#
#   sbatch run_generate.sh
#
#SBATCH --job-name=metagame
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=10G
#SBATCH --output=sbatch_logs/metaca_gen_%j.log
#SBATCH --time=00:30:00

set -e
mkdir -p sbatch_logs
hostname; pwd; date

conda run -n metaconceptattention python generate.py

date

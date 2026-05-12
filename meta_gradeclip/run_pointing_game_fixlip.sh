#!/bin/bash
#SBATCH --job-name=metagame
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=32G
#SBATCH --output=sbatch_logs/pointing_game_fixlip_%j.log
#SBATCH --time=00-11:00:00

set -e
mkdir -p sbatch_logs
hostname; pwd; date

conda run -n metagradeclip python pointing_game_fixlip.py \
    --model_name $1 \
    --path_input $2 \
    --path_output $3 \
    --class_labels $4 \
    --budget $5 \
    --batch_size $6 \
    --random_state $7

date

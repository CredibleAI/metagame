#!/bin/bash
# Submit one sbatch per dataset. Aggregates after the last shard finishes.
#
#   sbatch run_main.sh                              # MS COCO (10 shards, default)
#   DATASET=pascalvoc    sbatch --array=0-3  run_main.sh
#   DATASET=imagenetseg  sbatch --array=0-9  run_main.sh
#
# Tunable env vars (defaults in []):
#   DATASET         = mscoco | pascalvoc | imagenetseg   [mscoco]
#   SHARD_N         = number of array shards             [10]
#   TOTAL_PLAYERS   = exact player count per image       [20]
#                     (0 disables padding, image-conditional)
#   N_DISTRACTORS   = variable-size: per-image total =   [unset]
#                     n_present + N (overrides TOTAL_PLAYERS)
#   PROMPT          = set to 1 to feed FLUX an artificial []
#                     'a {cls1}, a {cls2}, …' prompt
#                     (paper Fig 13 'with artificial prompt')
#   DECOUPLED       = set to 1 to disable cross-concept   []
#                     attention (paper Fig 13 ablation)
#   LAYERS          = '14 15 16 17 18' for last-5-layers  [unset]
#                     ablation (default = last 10)
#   OUTPUT_DIR      = override default output dir         [unset]
#
#SBATCH --job-name=metagame
#SBATCH --gres=gpu:1
#SBATCH --array=0-9
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=10G
#SBATCH --output=sbatch_logs/metaca_%A_%a.log
#SBATCH --time=00-02:30:00

set -e
mkdir -p sbatch_logs
hostname; pwd; date

DATASET="${DATASET:-mscoco}"
SHARD_N="${SHARD_N:-10}"
SHARD_I="${SLURM_ARRAY_TASK_ID:-0}"

ARGS=()
[ -n "$OUTPUT_DIR" ]    && ARGS+=(--output-dir "$OUTPUT_DIR")
[ -n "$TOTAL_PLAYERS" ] && ARGS+=(--total-players "$TOTAL_PLAYERS")
[ -n "$N_DISTRACTORS" ] && ARGS+=(--n-distractors "$N_DISTRACTORS")
[ -n "$PROMPT" ]        && ARGS+=(--prompt)
[ -n "$DECOUPLED" ]     && ARGS+=(--decoupled)
[ -n "$LAYERS" ]        && ARGS+=(--layer-indices $LAYERS)

echo "=== dataset=$DATASET  shard=$SHARD_I/$SHARD_N  ${ARGS[*]} ==="

conda run -n metaconceptattention python main.py \
    --dataset "$DATASET" \
    --shard-i "$SHARD_I" --shard-n "$SHARD_N" \
    --device cuda:0 \
    "${ARGS[@]}"

# Aggregate is idempotent — whichever shard finishes last produces metrics.json.
# Earlier shards' aggregates are partial and get overwritten.
date
echo "=== attempting aggregate ==="
set +e
conda run -n metaconceptattention python main.py \
    --dataset "$DATASET" \
    "${ARGS[@]}" \
    --aggregate
set -e

date

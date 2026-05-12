#!/bin/bash
# sbatch run_main.sh google/gemma-3-4b-it
# sbatch --array=0-20 run_main.sh google/gemma-3-4b-it
#SBATCH --job-name=metagame
#SBATCH --gpus=1
#SBATCH --array=0-20%6
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --output=sbatch_logs/metagame_llms_%A_%a.log
#SBATCH --time=01-00:00:00

set -e
mkdir -p sbatch_logs
hostname; pwd; date

MODEL_ID="${1:-google/gemma-3-4b-it}"
MAX_NEW_TOKENS="${2:-100}"
PROMPTS_JSON="${3:-prompts.json}"
BATCH_SIZE="${4:-}"   # empty -> use COALITION_BATCH_SIZE_BY_MODEL default
IDX="${SLURM_ARRAY_TASK_ID:-0}"
echo "Model: $MODEL_ID | max_new_tokens: $MAX_NEW_TOKENS | prompts: $PROMPTS_JSON | batch: ${BATCH_SIZE:-default} | idx: $IDX"

DOMAIN=$(conda run -n metaattnlrp python -c "import json; print(json.load(open('$PROMPTS_JSON'))[$IDX]['domain'])")
PROMPT=$(conda run -n metaattnlrp python -c "import json; print(json.load(open('$PROMPTS_JSON'))[$IDX]['prompt'])")
echo "=== [$IDX] $DOMAIN ==="

EXTRA_ARGS=()
if [ -n "$BATCH_SIZE" ]; then
    EXTRA_ARGS+=(--coalition-batch-size "$BATCH_SIZE")
fi

conda run -n metaattnlrp python main.py \
    --model-id "$MODEL_ID" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --run-name "$DOMAIN" \
    --prompt "$PROMPT" \
    "${EXTRA_ARGS[@]}"

date

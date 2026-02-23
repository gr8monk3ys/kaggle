#!/usr/bin/env bash
# Kaggle Portfolio Manager
# Usage: ./manage.sh [command] [options]
#
# Commands:
#   status     - Show all notebooks/datasets and their Kaggle status
#   push-all   - Push all notebooks and datasets to Kaggle
#   push-nb    - Push all notebooks
#   push-ds    - Push all datasets
#   push       - Push a specific directory (e.g., ./manage.sh push med-gemma-challenge)
#   votes      - Show vote counts for all content
#   competitions - List active medal-eligible competitions
#   scorecard  - Generate medal operations scorecard from tracker
#   weekly-plan - Generate weekly execution plan from tracker
#   pace       - Generate velocity/ETA pace analysis from snapshots
#   sync       - Sync tracker metrics from live Kaggle CLI data
#   sync-template - Generate CSV templates and export helper for offline sync
#   doctor     - Run preflight checks for tracker/env/sync readiness
#   quality    - Score notebook quality and enforce minimum threshold

set -euo pipefail
export PATH="$HOME/.local/bin:$HOME/Library/Python/3.9/bin:$HOME/Library/Python/3.10/bin:$HOME/Library/Python/3.11/bin:$PATH"

KAGGLE_DIR="$(cd "$(dirname "$0")" && pwd)"
KAGGLE_CREDENTIALS_DEFAULT="${HOME}/.kaggle/kaggle.json"
KAGGLE_CREDENTIALS_LOCAL="${KAGGLE_DIR}/kaggle.json"

# Notebook directories (contain kernel-metadata.json with code_file)
NOTEBOOK_DIRS=(
    "akkadian-translation"
    "attention-guide"
    "competition-template"
    "digit-recognizer"
    "eda-tutorial"
    "ensemble-stacking"
    "feature-engineering"
    "financial-analysis"
    "fraud-detection"
    "graph-neural-networks"
    "house-prices"
    "image-segmentation"
    "llm-finetuning"
    "med-gemma-challenge"
    "nlp-disaster-tweets"
    "nlp-text-classification"
    "optuna-guide"
    "rag-from-scratch"
    "shap-explainability"
    "spaceship-titanic"
    "store-sales-forecasting"
    "timeseries-transformers"
    "titanic-ultimate"
    "vesuvius-surface"
)

# Dataset directories (contain dataset-metadata.json)
DATASET_DIRS=(
    "datasets/ai-research-trends"
    "datasets/credit-card-fraud"
    "datasets/ecommerce-behavior"
    "datasets/github-repo-metrics"
    "datasets/job-postings"
    "datasets/mental-health-tech"
    "datasets/ml-interview-qa"
    "datasets/programming-benchmarks"
    "datasets/spotify-tracks"
    "datasets/student-performance"
)

color_green='\033[0;32m'
color_red='\033[0;31m'
color_yellow='\033[0;33m'
color_blue='\033[0;34m'
color_reset='\033[0m'

usage() {
    cat <<EOF
Usage: $0 {status|push-all|push-nb|push-ds|push|votes|competitions|scorecard|weekly-plan|pace|sync|sync-template|doctor|quality|help}

Commands:
  status        Show notebooks/datasets and Kaggle account status
  push-all      Push all notebooks and datasets
  push-nb       Push all notebooks
  push-ds       Push all datasets
  push <dir>    Push a specific notebook/dataset directory
  votes         Show vote counts for all content
  competitions  List active medal-eligible competitions
  scorecard     Generate medal operations scorecard report
  weekly-plan   Generate weekly execution plan report
  pace          Generate medal progress pace analysis report
  sync          Sync tracker metrics from live Kaggle CLI data
  sync-template Generate CSV templates + export helper for offline sync
  doctor        Run preflight checks (tracker, sync inputs, environment)
  quality       Score notebook quality against rubric
  help          Show this message
EOF
}

require_kaggle_cli() {
    if ! command -v kaggle >/dev/null 2>&1; then
        echo "Error: kaggle CLI not found. Install it with: pip install kaggle" >&2
        exit 1
    fi
}

require_kaggle_credentials() {
    if [[ -f "$KAGGLE_CREDENTIALS_DEFAULT" || -f "$KAGGLE_CREDENTIALS_LOCAL" ]]; then
        return
    fi
    cat >&2 <<EOF
Error: Kaggle credentials not found.
Create ${KAGGLE_CREDENTIALS_DEFAULT} (recommended) and run:
  chmod 600 ${KAGGLE_CREDENTIALS_DEFAULT}
EOF
    exit 1
}

ensure_kaggle_ready() {
    require_kaggle_cli
    require_kaggle_credentials
}

cmd_status() {
    echo -e "${color_blue}=== Kaggle Portfolio Status ===${color_reset}"
    echo ""
    echo -e "${color_yellow}Notebooks:${color_reset}"
    kaggle kernels list --mine --page-size 50 2>/dev/null | head -30
    echo ""
    echo -e "${color_yellow}Datasets:${color_reset}"
    kaggle datasets list -m 2>/dev/null | head -20
    echo ""
    echo -e "${color_yellow}Local directories:${color_reset}"
    echo "  Notebooks: ${#NOTEBOOK_DIRS[@]}"
    echo "  Datasets:  ${#DATASET_DIRS[@]}"
}

cmd_push_all() {
    cmd_push_nb
    cmd_push_ds
}

cmd_push_nb() {
    echo -e "${color_blue}=== Pushing All Notebooks ===${color_reset}"
    local success=0
    local failed=0
    for dir in "${NOTEBOOK_DIRS[@]}"; do
        local path="$KAGGLE_DIR/$dir"
        if [[ -f "$path/kernel-metadata.json" ]]; then
            echo -n "  Pushing $dir... "
            if output=$(kaggle kernels push -p "$path" 2>&1); then
                echo -e "${color_green}OK${color_reset}"
                ((success += 1))
            else
                echo -e "${color_red}FAILED${color_reset}: $output"
                ((failed += 1))
            fi
        else
            echo -e "  ${color_yellow}SKIP${color_reset} $dir (no kernel-metadata.json)"
        fi
    done
    echo ""
    echo -e "Results: ${color_green}$success succeeded${color_reset}, ${color_red}$failed failed${color_reset}"
}

cmd_push_ds() {
    echo -e "${color_blue}=== Pushing All Datasets ===${color_reset}"
    local success=0
    local failed=0
    for dir in "${DATASET_DIRS[@]}"; do
        local path="$KAGGLE_DIR/$dir"
        if [[ -f "$path/dataset-metadata.json" ]]; then
            echo -n "  Pushing $dir... "
            if output=$(kaggle datasets version -p "$path" -m "Updated content" --dir-mode zip 2>&1); then
                echo -e "${color_green}UPDATED${color_reset}"
                ((success += 1))
            elif output=$(kaggle datasets create -p "$path" --dir-mode zip 2>&1); then
                echo -e "${color_green}CREATED${color_reset}"
                ((success += 1))
            else
                echo -e "${color_red}FAILED${color_reset}: $output"
                ((failed += 1))
            fi
        else
            echo -e "  ${color_yellow}SKIP${color_reset} $dir (no dataset-metadata.json)"
        fi
    done
    echo ""
    echo -e "Results: ${color_green}$success succeeded${color_reset}, ${color_red}$failed failed${color_reset}"
}

cmd_push() {
    local target="$1"
    local path="$KAGGLE_DIR/$target"
    if [[ -f "$path/kernel-metadata.json" ]]; then
        echo "Pushing notebook: $target"
        kaggle kernels push -p "$path"
    elif [[ -f "$path/dataset-metadata.json" ]]; then
        echo "Pushing dataset: $target"
        kaggle datasets version -p "$path" -m "Updated content" --dir-mode zip \
            || kaggle datasets create -p "$path" --dir-mode zip
    else
        echo "Error: No metadata found in $path"
        exit 1
    fi
}

cmd_votes() {
    echo -e "${color_blue}=== Vote Counts ===${color_reset}"
    echo ""
    echo -e "${color_yellow}Notebooks:${color_reset}"
    kaggle kernels list --mine --page-size 50 2>/dev/null \
        | tail -n +3 \
        | awk '{print $NF, $0}' \
        | sort -rn \
        | awk '{$1=""; print}'
    echo ""
    echo -e "${color_yellow}Datasets:${color_reset}"
    kaggle datasets list -m 2>/dev/null | tail -n +3
}

cmd_competitions() {
    echo -e "${color_blue}=== Active Medal-Eligible Competitions ===${color_reset}"
    kaggle competitions list --sort-by latestDeadline --category featured --page-size 10 2>/dev/null
    echo ""
    kaggle competitions list --sort-by latestDeadline --category research --page-size 10 2>/dev/null
}

cmd_scorecard() {
    python3 medal_ops.py scorecard "$@"
}

cmd_weekly_plan() {
    python3 medal_ops.py weekly-plan "$@"
}

cmd_pace() {
    python3 medal_ops.py pace "$@"
}

cmd_sync() {
    python3 medal_ops.py sync "$@"
}

cmd_sync_template() {
    python3 medal_ops.py sync-template "$@"
}

cmd_doctor() {
    python3 medal_ops.py doctor "$@"
}

cmd_quality() {
    python3 notebook_quality.py "$@"
}

# Main dispatcher
case "${1:-status}" in
    status)
        ensure_kaggle_ready
        cmd_status
        ;;
    push-all)
        ensure_kaggle_ready
        cmd_push_all
        ;;
    push-nb)
        ensure_kaggle_ready
        cmd_push_nb
        ;;
    push-ds)
        ensure_kaggle_ready
        cmd_push_ds
        ;;
    push)
        ensure_kaggle_ready
        cmd_push "${2:?Usage: $0 push <directory>}"
        ;;
    votes)
        ensure_kaggle_ready
        cmd_votes
        ;;
    competitions)
        ensure_kaggle_ready
        cmd_competitions
        ;;
    scorecard)
        cmd_scorecard "${@:2}"
        ;;
    weekly-plan)
        cmd_weekly_plan "${@:2}"
        ;;
    pace)
        cmd_pace "${@:2}"
        ;;
    sync)
        cmd_sync "${@:2}"
        ;;
    sync-template)
        cmd_sync_template "${@:2}"
        ;;
    doctor)
        cmd_doctor "${@:2}"
        ;;
    quality)
        cmd_quality "${@:2}"
        ;;
    help|-h|--help)
        usage
        ;;
    *)
        echo "Unknown command: $1"
        usage
        exit 1
        ;;
esac

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

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

KAGGLE_DIR="$(cd "$(dirname "$0")" && pwd)"

# Notebook directories (contain kernel-metadata.json with code_file)
NOTEBOOK_DIRS=(
    "akkadian-translation"
    "attention-guide"
    "competition-template"
    "eda-tutorial"
    "ensemble-stacking"
    "feature-engineering"
    "financial-analysis"
    "fraud-detection"
    "graph-neural-networks"
    "image-segmentation"
    "llm-finetuning"
    "med-gemma-challenge"
    "rag-from-scratch"
    "timeseries-transformers"
    "vesuvius-surface"
)

# Dataset directories (contain dataset-metadata.json)
DATASET_DIRS=(
    "datasets/ai-research-trends"
    "datasets/ecommerce-behavior"
    "datasets/github-repo-metrics"
    "datasets/ml-interview-qa"
    "datasets/programming-benchmarks"
)

color_green='\033[0;32m'
color_red='\033[0;31m'
color_yellow='\033[0;33m'
color_blue='\033[0;34m'
color_reset='\033[0m'

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
                ((success++))
            else
                echo -e "${color_red}FAILED${color_reset}: $output"
                ((failed++))
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
            if output=$(kaggle datasets create -p "$path" --dir-mode zip 2>&1); then
                echo -e "${color_green}OK${color_reset}"
                ((success++))
            elif output2=$(kaggle datasets version -p "$path" -m "Updated content" --dir-mode zip 2>&1); then
                echo -e "${color_green}UPDATED${color_reset}"
                ((success++))
            else
                echo -e "${color_red}FAILED${color_reset}: $output"
                ((failed++))
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
        kaggle datasets version -p "$path" -m "Updated content" --dir-mode zip 2>/dev/null \
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

# Main dispatcher
case "${1:-status}" in
    status)       cmd_status ;;
    push-all)     cmd_push_all ;;
    push-nb)      cmd_push_nb ;;
    push-ds)      cmd_push_ds ;;
    push)         cmd_push "${2:?Usage: $0 push <directory>}" ;;
    votes)        cmd_votes ;;
    competitions) cmd_competitions ;;
    *)
        echo "Unknown command: $1"
        echo "Usage: $0 {status|push-all|push-nb|push-ds|push|votes|competitions}"
        exit 1
        ;;
esac

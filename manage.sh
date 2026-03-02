#!/usr/bin/env bash
# Kaggle Portfolio Manager
# Usage: ./manage.sh [command] [options]
#
# Commands:
#   status          - Show all notebooks/datasets and their Kaggle status
#   push-all        - Push all notebooks and datasets to Kaggle
#   push-nb         - Push all notebooks
#   push-ds         - Push all datasets
#   push            - Push a specific directory (e.g., ./manage.sh push med-gemma-challenge)
#   validate        - Validate all kernel-metadata.json files before pushing
#   votes           - Show vote counts with medal threshold dashboard
#   competitions    - List active medal-eligible competitions
#   link-competition - Link a notebook to a competition (e.g., ./manage.sh link-competition spaceship-titanic spaceship-titanic)
#   scorecard       - Generate medal operations scorecard from tracker
#   weekly-plan     - Generate weekly execution plan from tracker
#   pace            - Generate velocity/ETA pace analysis from snapshots
#   sync            - Sync tracker metrics from live Kaggle CLI data
#   sync-template   - Generate CSV templates and export helper for offline sync
#   doctor          - Run preflight checks for tracker/env/sync readiness
#   quality         - Score notebook quality and enforce minimum threshold
#   dataset-usability - Score dataset usability and generate reports
#   usability-tracker - Run daily live usability tracker with 0.8 gate, 1.0 target
#   campaign-pack   - Build multi-channel dataset promotion campaign + queue
#   campaign-run    - Execute campaign queue (show/claim/complete + runbook)
#   usability-benchmark - Benchmark local datasets against public 1.0-usability exemplars
#   publish-datasets - Publish draft datasets with quality gate checks
#   auth-doctor     - Validate Kaggle credentials and upload auth
#   draft-set       - Update a queued draft status/priority/deadline
#   dataset-ui-sync - Sync dataset UI-only metadata fields using Playwright (supports throttling)
#   stale-content   - Detect stale notebooks, datasets, and outdated library versions
#   build-explore-notebooks - Generate rich EDA explore notebooks for datasets
#   create-competition-entry - Scaffold a new competition entry from a slug
#   metadata-tracker - Track metadata changes vs vote deltas over time

set -euo pipefail
export PATH="$HOME/.local/bin:$HOME/Library/Python/3.9/bin:$HOME/Library/Python/3.10/bin:$HOME/Library/Python/3.11/bin:$PATH"

KAGGLE_DIR="$(cd "$(dirname "$0")" && pwd)"
KAGGLE_CREDENTIALS_DEFAULT="${HOME}/.kaggle/kaggle.json"
KAGGLE_CREDENTIALS_LOCAL="${KAGGLE_DIR}/kaggle.json"

# Auto-discover notebook directories: any dir with kernel-metadata.json, excluding datasets/
discover_notebook_dirs() {
    while IFS= read -r meta; do
        dir="$(dirname "$meta")"
        # Exclude dataset explore notebooks from the notebook list
        case "$dir" in
            "$KAGGLE_DIR/datasets/"*) ;;
            *) echo "${dir#$KAGGLE_DIR/}" ;;
        esac
    done < <(find "$KAGGLE_DIR" -maxdepth 3 -name "kernel-metadata.json" | sort)
}

# Auto-discover dataset directories: any dir with dataset-metadata.json
discover_dataset_dirs() {
    while IFS= read -r meta; do
        dir="$(dirname "$meta")"
        echo "${dir#$KAGGLE_DIR/}"
    done < <(find "$KAGGLE_DIR" -maxdepth 3 -name "dataset-metadata.json" | sort)
}

# Build arrays from discovery (bash 3.2-compatible; macOS ships bash 3)
NOTEBOOK_DIRS=()
while IFS= read -r _dir; do
    NOTEBOOK_DIRS+=("$_dir")
done < <(discover_notebook_dirs)

DATASET_DIRS=()
while IFS= read -r _dir; do
    DATASET_DIRS+=("$_dir")
done < <(discover_dataset_dirs)

color_green='\033[0;32m'
color_red='\033[0;31m'
color_yellow='\033[0;33m'
color_blue='\033[0;34m'
color_reset='\033[0m'

usage() {
    cat <<EOF
Usage: $0 {status|push-all|push-nb|push-ds|push|validate|votes|competitions|link-competition|scorecard|weekly-plan|pace|sync|sync-template|doctor|quality|dataset-usability|usability-tracker|campaign-pack|campaign-run|campaign-execute|usability-benchmark|publish-datasets|auth-doctor|build-all|optimize-datasets|post-discussion|draft-ops|draft-set|dataset-ui-sync|promote-notebooks|scout|stale-content|build-explore-notebooks|create-competition-entry|metadata-tracker|help}

Commands:
  status                    Show notebooks/datasets and Kaggle account status
  push-all                  Push all notebooks and datasets
  push-nb                   Push all notebooks
  push-ds                   Push all datasets
  push <dir>                Push a specific notebook/dataset directory
  validate [--pre-push]     Validate all kernel-metadata.json files (JSON, required fields, file existence)
  votes                     Show vote counts with bronze/silver/gold medal threshold dashboard
  competitions              List active medal-eligible competitions
  link-competition <dir> <slug>  Add competition_sources to a notebook and re-push
  scorecard                 Generate medal operations scorecard report
  weekly-plan               Generate weekly execution plan report
  pace                      Generate medal progress pace analysis report
  sync                      Sync tracker metrics from live Kaggle CLI data
  sync-template             Generate CSV templates + export helper for offline sync
  doctor                    Run preflight checks (tracker, sync inputs, environment)
  quality                   Score notebook quality against rubric
  dataset-usability         Score dataset usability and generate reports
  usability-tracker         Daily live tracker with threshold alerts and ranked action queue
  campaign-pack             Generate multi-channel promotion campaign pack + queue
  campaign-run              Execute campaign queue (show/claim/complete + runbook export)
  usability-benchmark       Benchmark local datasets against public high-usability exemplars
  publish-datasets [--apply] [--all] [--min-score N] [--owner OWNER] [--max-items N]
                            Publish datasets through draft/live + quality gates
  auth-doctor [--strict] [--expected-owner OWNER]
                            Validate Kaggle credentials, owner alignment, and upload auth
  build-all [--stale-only] [--push] [--validate-only] [--min-score N]
                            Build all notebooks with build_notebook.py scripts
  optimize-datasets [--push]  Generate README.md + improve dataset descriptions
  post-discussion [--dry-run|--init|--schedule-weeks N]
                            Post next queued discussion draft or rebuild queue window
  draft-ops                  Show draft backlog stage counts + priority queue
  draft-set <id> [--status STATUS] [--priority PRIORITY] [--deadline YYYY-MM-DD|--clear-deadline] [--schedule-weeks N]
                            Update draft metadata and rebalance queue schedule window
  dataset-ui-sync [--apply] [--headed] [--dataset <dir>] [--dataset-ref <owner/slug>]
                  [--max-datasets N] [--sleep-between-datasets-s SEC]
                            Sync Kaggle UI-only dataset sections
                            (Authors/Coverage/DOI/Provenance/Citations/License/Update Frequency/File Info)
                            Use throttling flags to reduce request bursts and avoid Kaggle rate-limit toasts.
  promote-notebooks [--auto]   Generate notebook promotion plan for competition forums
  scout [--update]          Scout active competitions ranked by medal opportunity
  stale-content [--max-nb-age N] [--max-ds-age N]
                             Detect stale notebooks, datasets, and outdated library versions
  build-explore-notebooks [--push]
                             Generate rich EDA explore notebooks for all datasets
  create-competition-entry <slug> [--gpu] [--push]
                             Scaffold a new competition entry from a competition slug
  metadata-tracker <snapshot|annotate|report> [args...]
                             Track metadata changes vs vote deltas over time
                             Detect stale notebooks, datasets, and outdated library versions
  build-explore-notebooks [--push]
                             Generate rich EDA explore notebooks for all datasets
  create-competition-entry <slug> [--gpu] [--push]
                             Scaffold a new competition entry from a competition slug
  metadata-tracker <snapshot|annotate|report> [args...]
                             Track metadata changes vs vote deltas over time
  help                      Show this message
EOF
}

require_kaggle_cli() {
    if command -v kaggle >/dev/null 2>&1; then
        return
    fi
    if python3 -m kaggle.cli --version >/dev/null 2>&1; then
        return
    fi
    echo "Error: kaggle CLI not found. Install it with: pip install kaggle" >&2
    exit 1
}

kaggle_cli() {
    if command -v kaggle >/dev/null 2>&1; then
        kaggle "$@"
    else
        python3 -m kaggle.cli "$@"
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
    kaggle_cli kernels list --mine --page-size 50 2>/dev/null | head -30
    echo ""
    echo -e "${color_yellow}Datasets:${color_reset}"
    kaggle_cli datasets list -m 2>/dev/null | head -20
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
    echo -e "${color_yellow}Running pre-push validation...${color_reset}"
    if ! cmd_validate; then
        echo -e "${color_red}Fix validation errors before pushing.${color_reset}"
        return 1
    fi
    echo ""
    local success=0
    local failed=0
    for dir in "${NOTEBOOK_DIRS[@]}"; do
        local path="$KAGGLE_DIR/$dir"
        if [[ -f "$path/kernel-metadata.json" ]]; then
            echo -n "  Pushing $dir... "
            if output=$(kaggle_cli kernels push -p "$path" 2>&1); then
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
    echo -e "${color_yellow}Running pre-push validation...${color_reset}"
    if ! cmd_validate; then
        echo -e "${color_red}Fix validation errors before pushing.${color_reset}"
        return 1
    fi
    echo ""
    local success=0
    local failed=0
    for dir in "${DATASET_DIRS[@]}"; do
        local path="$KAGGLE_DIR/$dir"
        if [[ -f "$path/dataset-metadata.json" ]]; then
            echo -n "  Pushing $dir... "
            if output=$(kaggle_cli datasets version -p "$path" -m "Updated content" --dir-mode zip 2>&1); then
                echo -e "${color_green}UPDATED${color_reset}"
                ((success += 1))
            elif output=$(kaggle_cli datasets create -p "$path" --dir-mode zip 2>&1); then
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
        kaggle_cli kernels push -p "$path"
    elif [[ -f "$path/dataset-metadata.json" ]]; then
        echo "Pushing dataset: $target"
        kaggle_cli datasets version -p "$path" -m "Updated content" --dir-mode zip \
            || kaggle_cli datasets create -p "$path" --dir-mode zip
    else
        echo "Error: No metadata found in $path"
        exit 1
    fi
}

cmd_validate() {
    local pre_push="${1:-}"
    local errors=0
    local checked=0

    echo -e "${color_blue}=== Validating kernel-metadata.json files ===${color_reset}"
    echo ""

    while IFS= read -r meta; do
        local rel="${meta#$KAGGLE_DIR/}"
        local dir
        dir="$(dirname "$meta")"
        ((checked += 1))

        # 1. JSON syntax
        if ! python3 -c "import json,sys; json.load(open('$meta'))" 2>/dev/null; then
            echo -e "  ${color_red}FAIL${color_reset} $rel — invalid JSON"
            ((errors += 1))
            continue
        fi

        local id title code_file
        id=$(python3 -c "import json; d=json.load(open('$meta')); print(d.get('id',''))" 2>/dev/null)
        title=$(python3 -c "import json; d=json.load(open('$meta')); print(d.get('title',''))" 2>/dev/null)
        code_file=$(python3 -c "import json; d=json.load(open('$meta')); print(d.get('code_file',''))" 2>/dev/null)

        local file_errors=()

        # 2. Required fields
        [[ -z "$id" ]]        && file_errors+=("missing 'id'")
        [[ -z "$title" ]]     && file_errors+=("missing 'title'")
        [[ -z "$code_file" ]] && file_errors+=("missing 'code_file'")

        # 3. Title length (6–70 chars)
        local title_len=${#title}
        [[ $title_len -gt 0 && ($title_len -lt 6 || $title_len -gt 70) ]] \
            && file_errors+=("title length $title_len (must be 6-70)")

        # 4. code_file must exist in the same directory
        if [[ -n "$code_file" && ! -f "$dir/$code_file" ]]; then
            file_errors+=("code_file '$code_file' not found in $(basename "$dir")/")
        fi

        # 5. No credentials in the metadata file
        if grep -qiE '(password|secret|api_key|KGAT_|kaggle_token)' "$meta" 2>/dev/null; then
            file_errors+=("possible credential in metadata — review before pushing")
        fi

        if [[ ${#file_errors[@]} -eq 0 ]]; then
            echo -e "  ${color_green}OK${color_reset}   $rel"
        else
            echo -e "  ${color_red}FAIL${color_reset} $rel"
            for err in "${file_errors[@]}"; do
                echo -e "       ${color_yellow}→${color_reset} $err"
            done
            ((errors += 1))
        fi
    done < <(find "$KAGGLE_DIR" -maxdepth 3 -name "kernel-metadata.json" | sort)

    echo ""
    echo "Checked $checked files — ${errors} error(s)"

    if [[ $errors -gt 0 ]]; then
        echo -e "${color_red}Validation FAILED. Fix errors before pushing.${color_reset}"
        return 1
    fi
    echo -e "${color_green}All metadata files valid.${color_reset}"
    return 0
}

cmd_votes() {
    # Medal thresholds for Kaggle notebooks (approximate community benchmarks)
    local BRONZE_THRESHOLD=5
    local SILVER_THRESHOLD=20
    local GOLD_THRESHOLD=50

    echo -e "${color_blue}=== Vote Counts & Medal Threshold Dashboard ===${color_reset}"
    echo ""
    echo -e "${color_yellow}Medal thresholds:${color_reset}  Bronze ≥${BRONZE_THRESHOLD}  Silver ≥${SILVER_THRESHOLD}  Gold ≥${GOLD_THRESHOLD}"
    echo ""

    # Fetch raw CSV from Kaggle CLI
    local raw
    if ! raw=$(kaggle_cli kernels list --mine --page-size 50 --csv 2>/dev/null); then
        echo -e "${color_red}Failed to fetch kernel list from Kaggle.${color_reset}"
        return 1
    fi

    # Parse with Python for reliable CSV handling and formatting
    RAW_CSV="$raw" python3 - "$BRONZE_THRESHOLD" "$SILVER_THRESHOLD" "$GOLD_THRESHOLD" <<'PYEOF'
import csv, io, os, sys

bronze_t = int(sys.argv[1])
silver_t = int(sys.argv[2])
gold_t   = int(sys.argv[3])

GREEN  = "\033[0;32m"
YELLOW = "\033[0;33m"
RED    = "\033[0;31m"
CYAN   = "\033[0;36m"
RESET  = "\033[0m"

raw_lines = os.environ.get("RAW_CSV", "")
rows = list(csv.DictReader(io.StringIO(raw_lines)))

if not rows:
    print("No kernels found.")
    sys.exit(0)

# Find the votes column (different kaggle CLI versions use different names)
vote_col = next(
    (k for k in (rows[0] if rows else {}) if "vote" in k.lower() or "upvote" in k.lower()),
    None,
)
ref_col = next(
    (k for k in (rows[0] if rows else {}) if "ref" in k.lower() or "title" in k.lower()),
    "ref",
)

def next_tier(votes):
    if votes < bronze_t: return ("Bronze", bronze_t)
    if votes < silver_t: return ("Silver", silver_t)
    if votes < gold_t:   return ("Gold",   gold_t)
    return ("GOLD+", gold_t)

def medal_color(votes):
    if votes >= gold_t:   return CYAN
    if votes >= silver_t: return GREEN
    if votes >= bronze_t: return YELLOW
    return RESET

print(f"{'Notebook':<45} {'Votes':>6}  {'Tier':<8} {'Next':<8} {'Gap':>4}")
print("-" * 78)

# Sort by votes descending, notebooks closest to next tier first within same tier
def sort_key(r):
    v = int(r.get(vote_col, 0) or 0) if vote_col else 0
    tier_name, tier_thresh = next_tier(v)
    gap = tier_thresh - v
    return (-v, gap)

for row in sorted(rows, key=sort_key):
    ref   = row.get(ref_col, row.get("ref", "unknown"))
    votes = int(row.get(vote_col, 0) or 0) if vote_col else 0
    tier_name, tier_thresh = next_tier(votes)
    gap = tier_thresh - votes

    # Current medal label
    if votes >= gold_t:   cur = "GOLD"
    elif votes >= silver_t: cur = "Silver"
    elif votes >= bronze_t: cur = "Bronze"
    else: cur = "—"

    col = medal_color(votes)
    # Highlight green if within 3 votes of next tier
    gap_color = GREEN if (0 < gap <= 3) else (YELLOW if (0 < gap <= 10) else RESET)

    name = str(ref).split("/")[-1][:44]
    print(f"{col}{name:<45}{RESET} {votes:>6}  {cur:<8} {tier_name:<8} {gap_color}{gap:>4}{RESET}")

print()
if not vote_col:
    print("(vote column not found in CSV — run: kaggle kernels list --mine --csv to inspect headers)")
PYEOF

    echo ""
    echo -e "${color_yellow}Datasets:${color_reset}"
    kaggle_cli datasets list -m 2>/dev/null | tail -n +3
}

cmd_link_competition() {
    local dir="${1:?Usage: $0 link-competition <notebook-dir> <competition-slug>}"
    local slug="${2:?Usage: $0 link-competition <notebook-dir> <competition-slug>}"
    local path="$KAGGLE_DIR/$dir"
    local meta="$path/kernel-metadata.json"

    if [[ ! -f "$meta" ]]; then
        echo -e "${color_red}Error: kernel-metadata.json not found in $dir${color_reset}" >&2
        exit 1
    fi

    echo "Linking $dir → competition '$slug' ..."

    # Update competition_sources via Python (handles existing array or missing key)
    python3 - "$meta" "$slug" <<'PYEOF'
import json, sys

meta_path = sys.argv[1]
slug      = sys.argv[2]

with open(meta_path, encoding="utf-8") as f:
    meta = json.load(f)

sources = meta.get("competition_sources", [])
if slug not in sources:
    sources.append(slug)
meta["competition_sources"] = sources

with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)
    f.write("\n")

print(f"competition_sources = {json.dumps(sources)}")
PYEOF

    echo "Pushing updated notebook ..."
    kaggle_cli kernels push -p "$path"
    echo -e "${color_green}Done. Remember to accept the competition rules on Kaggle.com if push fails.${color_reset}"
}

cmd_competitions() {
    echo -e "${color_blue}=== Active Medal-Eligible Competitions ===${color_reset}"
    kaggle_cli competitions list --sort-by latestDeadline --category featured --page-size 10 2>/dev/null
    echo ""
    kaggle_cli competitions list --sort-by latestDeadline --category research --page-size 10 2>/dev/null
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

cmd_dataset_usability() {
    python3 dataset_usability.py "$@"
}

cmd_usability_tracker() {
    python3 dataset_usability.py --live --daily-tracker --alert-under 0.8 --target-rating 1.0 --fail-on-live-alert "$@"
}

cmd_campaign_pack() {
    python3 campaign_pack.py "$@"
}

cmd_campaign_run() {
    python3 campaign_dispatcher.py "$@"
}

cmd_usability_benchmark() {
    python3 dataset_usability_benchmark.py "$@"
}

cmd_publish_datasets() {
    python3 dataset_publish_pipeline.py "$@"
}

cmd_auth_doctor() {
    python3 kaggle_auth_doctor.py "$@"
}

cmd_build_all() {
    python3 notebook_pipeline.py "$@"
}

cmd_dataset_ui_sync() {
    python3 pi-automation/scripts/dataset_metadata_sync.py "$@"
}

cmd_stale_content() {
    python3 stale_content_detector.py "$@"
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
    validate)
        cmd_validate "${2:-}"
        ;;
    votes)
        ensure_kaggle_ready
        cmd_votes
        ;;
    competitions)
        ensure_kaggle_ready
        cmd_competitions
        ;;
    link-competition)
        ensure_kaggle_ready
        cmd_link_competition "${2:-}" "${3:-}"
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
    dataset-usability)
        cmd_dataset_usability "${@:2}"
        ;;
    usability-tracker)
        cmd_usability_tracker "${@:2}"
        ;;
    campaign-pack)
        cmd_campaign_pack "${@:2}"
        ;;
    campaign-run)
        cmd_campaign_run "${@:2}"
        ;;
    usability-benchmark)
        ensure_kaggle_ready
        cmd_usability_benchmark "${@:2}"
        ;;
    publish-datasets)
        ensure_kaggle_ready
        cmd_publish_datasets "${@:2}"
        ;;
    auth-doctor)
        cmd_auth_doctor "${@:2}"
        ;;
    build-all)
        cmd_build_all "${@:2}"
        ;;
    optimize-datasets)
        python3 dataset_optimizer.py "${@:2}"
        ;;
    post-discussion)
        python3 discussion_scheduler.py "${@:2}"
        ;;
    draft-ops)
        python3 discussion_scheduler.py --ops-report "${@:2}"
        ;;
    draft-set)
        python3 discussion_scheduler.py --set-id "${2:?Usage: $0 draft-set <draft_id> [--status ...] [--priority ...] [--deadline YYYY-MM-DD|--clear-deadline] [--schedule-weeks N]}" "${@:3}"
        ;;
    dataset-ui-sync)
        cmd_dataset_ui_sync "${@:2}"
        ;;
    promote-notebooks)
        python3 notebook_promoter.py "${@:2}"
        ;;
    scout)
        python3 competition_scout.py "${@:2}"
        ;;
    stale-content)
        cmd_stale_content "${@:2}"
        ;;
    build-explore-notebooks)
        python3 dataset_explore_generator.py --all "${@:2}"
        ;;
    create-competition-entry)
        python3 competition_entry.py "${@:2}"
        ;;
    metadata-tracker)
        python3 metadata_tracker.py "${@:2}"
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

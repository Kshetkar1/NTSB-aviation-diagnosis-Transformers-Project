#!/usr/bin/env bash
set -euo pipefail

# Orchestration script to run multiple subagents in sequence or parallel
# Usage:
#   ./orchestrate_subagents.sh <config_file>
#   or
#   ./orchestrate_subagents.sh --role implementer --task T001 --prompt prompt1.txt --output out1.txt

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPAWN_SCRIPT="$SCRIPT_DIR/spawn_cli_subagent.sh"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
  echo -e "${GREEN}[INFO]${NC} $*" >&2
}

log_warn() {
  echo -e "${YELLOW}[WARN]${NC} $*" >&2
}

log_error() {
  echo -e "${RED}[ERROR]${NC} $*" >&2
}

# Run a single subagent
run_subagent() {
  local role="$1"
  local task_id="$2"
  local prompt_file="$3"
  local output_file="$4"
  local model="${5:-auto}"
  
  log_info "Running subagent: $role:$task_id"
  log_info "  Prompt: $prompt_file"
  log_info "  Output: $output_file"
  
  if MODEL="$model" "$SPAWN_SCRIPT" "$role" "$task_id" "$prompt_file" "$output_file"; then
    log_info "✓ Subagent $role:$task_id completed successfully"
    return 0
  else
    local exit_code=$?
    log_error "✗ Subagent $role:$task_id failed with exit code $exit_code"
    return $exit_code
  fi
}

# Run multiple subagents in sequence
run_sequence() {
  local config_file="$1"
  local failed=0
  
  while IFS='|' read -r role task_id prompt_file output_file model; do
    # Skip empty lines and comments
    [[ -z "$role" || "$role" =~ ^# ]] && continue
    
    # Trim whitespace
    role=$(echo "$role" | xargs)
    task_id=$(echo "$task_id" | xargs)
    prompt_file=$(echo "$prompt_file" | xargs)
    output_file=$(echo "$output_file" | xargs)
    model=$(echo "${model:-auto}" | xargs)
    
    if ! run_subagent "$role" "$task_id" "$prompt_file" "$output_file" "$model"; then
      failed=$((failed + 1))
      log_warn "Stopping sequence due to failure"
      break
    fi
  done < "$config_file"
  
  return $failed
}

# Run multiple subagents in parallel (if they have no dependencies)
run_parallel() {
  local config_file="$1"
  local pids=()
  local failed=0
  
  while IFS='|' read -r role task_id prompt_file output_file model; do
    # Skip empty lines and comments
    [[ -z "$role" || "$role" =~ ^# ]] && continue
    
    # Trim whitespace
    role=$(echo "$role" | xargs)
    task_id=$(echo "$task_id" | xargs)
    prompt_file=$(echo "$prompt_file" | xargs)
    output_file=$(echo "$output_file" | xargs)
    model=$(echo "${model:-auto}" | xargs)
    
    log_info "Starting parallel subagent: $role:$task_id"
    run_subagent "$role" "$task_id" "$prompt_file" "$output_file" "$model" &
    pids+=($!)
  done < "$config_file"
  
  # Wait for all processes and collect exit codes
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=$((failed + 1))
    fi
  done
  
  return $failed
}

# Main execution
if [[ $# -eq 0 ]]; then
  echo "Usage: $0 <config_file> [--parallel]"
  echo "   or: $0 --role <role> --task <task_id> --prompt <prompt_file> --output <output_file> [--model <model>]"
  echo ""
  echo "Config file format (CSV-like, pipe-separated):"
  echo "  role|task_id|prompt_file|output_file|model"
  echo "  implementer|T001|prompt1.txt|out1.txt|auto"
  echo "  verifier|T002|prompt2.txt|out2.txt|sonnet"
  exit 1
fi

# Single subagent mode
if [[ "$1" == "--role" ]]; then
  shift
  ROLE="$1"
  shift && [[ "$1" == "--task" ]] && shift
  TASK_ID="$1"
  shift && [[ "$1" == "--prompt" ]] && shift
  PROMPT_FILE="$1"
  shift && [[ "$1" == "--output" ]] && shift
  OUTPUT_FILE="$1"
  MODEL="${2:-auto}"
  if [[ "$MODEL" == "--model" ]]; then
    shift
    MODEL="$1"
  fi
  
  run_subagent "$ROLE" "$TASK_ID" "$PROMPT_FILE" "$OUTPUT_FILE" "$MODEL"
  exit $?
fi

# Config file mode
CONFIG_FILE="$1"
PARALLEL="${2:-}"

if [[ ! -f "$CONFIG_FILE" ]]; then
  log_error "Config file not found: $CONFIG_FILE"
  exit 1
fi

if [[ "$PARALLEL" == "--parallel" ]]; then
  log_info "Running subagents in parallel mode"
  run_parallel "$CONFIG_FILE"
else
  log_info "Running subagents in sequence mode"
  run_sequence "$CONFIG_FILE"
fi

exit $?
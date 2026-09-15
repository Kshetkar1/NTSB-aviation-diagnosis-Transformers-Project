#!/usr/bin/env bash
set -euo pipefail

ROLE="${1:-implementer}"          # e.g., implementer | verifier
TASK_ID="${2:-T000}"              # e.g., T003
PROMPT_FILE="${3:-/dev/stdin}"    # path to prompt.txt
OUTPUT_FILE="${4:-/dev/stdout}"   # path to output.txt
MODEL="${MODEL:-auto}"            # allow override: MODEL=sonnet

# Only create output directory if OUTPUT_FILE is not a special file
if [[ "$OUTPUT_FILE" != "/dev/stdout" && "$OUTPUT_FILE" != "/dev/stderr" ]]; then
  mkdir -p "$(dirname "$OUTPUT_FILE")"
fi

# Read prompt from file or stdin
if [[ "$PROMPT_FILE" == "/dev/stdin" ]]; then
  # Read from stdin directly
  PROMPT="$(cat)"
else
  # Validate file exists and is readable
  if [[ ! -r "$PROMPT_FILE" ]]; then
    echo "Error: Cannot read prompt file: $PROMPT_FILE" >&2
    exit 1
  fi
  PROMPT="$(cat "$PROMPT_FILE")"
fi

# Log role and task ID for debugging (if not writing to stdout)
if [[ "$OUTPUT_FILE" != "/dev/stdout" ]]; then
  echo "[$ROLE:$TASK_ID] Starting subagent..." >&2
fi

# Choose an available CLI entrypoint.
# Common possibilities include: cursor-agent, agent, or cursor (depending on installation).
# Build command arguments
CMD_ARGS=()
if [[ "$MODEL" != "auto" ]]; then
  CMD_ARGS+=(--model "$MODEL")
fi
CMD_ARGS+=(--print --output-format=text -f)  # -f = --force for non-interactive mode

if command -v cursor-agent >/dev/null 2>&1; then
  # Example-style invocation seen in common Cursor CLI usage patterns.
  cursor-agent "${CMD_ARGS[@]}" "$PROMPT" | tee "$OUTPUT_FILE"
elif command -v agent >/dev/null 2>&1; then
  # Some installations expose `agent` directly.
  agent "${CMD_ARGS[@]}" "$PROMPT" | tee "$OUTPUT_FILE"
elif command -v cursor >/dev/null 2>&1; then
  # Fallback: if cursor supports agent mode via subcommand.
  cursor agent "${CMD_ARGS[@]}" "$PROMPT" | tee "$OUTPUT_FILE"
else
  echo "No Cursor CLI found (cursor-agent/agent/cursor). Install/enable Cursor CLI first." | tee "$OUTPUT_FILE"
  exit 127
fi

# Cursor CLI Subagent Scripts

Scripts for spawning and orchestrating Cursor CLI subagents in non-interactive mode.

## Scripts

### `spawn_cli_subagent.sh`

Spawns a single Cursor CLI subagent.

**Usage:**
```bash
./spawn_cli_subagent.sh [ROLE] [TASK_ID] [PROMPT_FILE] [OUTPUT_FILE]
```

**Parameters:**
- `ROLE` (default: `implementer`): Role identifier (e.g., `implementer`, `verifier`, `reviewer`)
- `TASK_ID` (default: `T000`): Task identifier (e.g., `T001`, `T002`)
- `PROMPT_FILE` (default: `/dev/stdin`): Path to prompt file or `/dev/stdin` to read from stdin
- `OUTPUT_FILE` (default: `/dev/stdout`): Path to output file or `/dev/stdout` for stdout

**Environment Variables:**
- `MODEL` (default: `auto`): Model to use (e.g., `sonnet`, `gpt-5`). Set to `auto` to use default.

**Examples:**
```bash
# Read prompt from file, write output to file
./spawn_cli_subagent.sh implementer T001 prompt.txt output.txt

# Read prompt from stdin, write to stdout
echo "Your prompt here" | ./spawn_cli_subagent.sh implementer T001

# Use specific model
MODEL=sonnet ./spawn_cli_subagent.sh verifier T002 prompt.txt output.txt
```

### `orchestrate_subagents.sh`

Orchestrates multiple subagents, running them in sequence or parallel.

**Usage:**
```bash
# Using config file (sequence mode)
./orchestrate_subagents.sh config.txt

# Using config file (parallel mode)
./orchestrate_subagents.sh config.txt --parallel

# Single subagent mode
./orchestrate_subagents.sh --role implementer --task T001 --prompt prompt.txt --output output.txt [--model sonnet]
```

**Config File Format:**

Pipe-separated values (CSV-like):
```
role|task_id|prompt_file|output_file|model
implementer|T001|prompt1.txt|out1.txt|auto
verifier|T002|prompt2.txt|out2.txt|sonnet
```

Lines starting with `#` are comments and will be ignored.

**Example Config File:**
```bash
# RALF workflow: implement then verify
implementer|T001|.cursor/state/ralf/prompts/implement.txt|.cursor/state/ralf/logs/implement.log|auto
verifier|T002|.cursor/state/ralf/prompts/verify.txt|.cursor/state/ralf/logs/verify.log|auto
reviewer|T003|.cursor/state/ralf/prompts/review.txt|.cursor/state/ralf/logs/review.log|sonnet
```

## Integration with RALF Workflow

These scripts are designed to work with the RALF (Loop Until Truth) workflow:

1. **Implement:** `spawn_cli_subagent.sh implementer T001 implement_prompt.txt implement_output.txt`
2. **Verify:** Run tests/lint gates
3. **Review:** `spawn_cli_subagent.sh reviewer T002 review_prompt.txt review_output.txt`
4. **Iterate:** If any step fails, loop back

See `.cursor/skills/ralf/SKILL.md` for the full workflow.

## Requirements

- Cursor CLI must be installed and available as one of:
  - `cursor-agent` command
  - `agent` command  
  - `cursor agent` subcommand (most common)

The script auto-detects which is available.

## Error Handling

- Scripts exit with non-zero status on failure
- Error messages are written to stderr
- Output files are only created if the subagent succeeds
- In sequence mode, orchestration stops on first failure

## Notes

- Subagents run in non-interactive mode (`--print --output-format=text -f`)
- All output is logged to both stdout and the specified output file (via `tee`)
- Role and task ID are logged to stderr for debugging
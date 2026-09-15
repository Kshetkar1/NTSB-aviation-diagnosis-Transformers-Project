#!/usr/bin/env python3
"""RALF Stop Hook -- auto-continues the RALF loop if tasks remain.

Cursor calls this hook via .cursor/hooks.json on the "stop" event.
It reads JSON from stdin with fields like conversation_id, status, loop_count.

Safety:
- Only active when .cursor/state/ralf/ENABLE_STOP_HOOK exists.
  Create this file to enable: touch .cursor/state/ralf/ENABLE_STOP_HOOK
  Remove it to disable:       rm .cursor/state/ralf/ENABLE_STOP_HOOK
- Stops after MAX_ITERATIONS to prevent infinite loops.
"""

import json
import sys
from pathlib import Path

MAX_ITERATIONS = 8


def main():
    # Read JSON payload from stdin (Cursor passes conversation metadata)
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            print(json.dumps({}))
            return
        payload = json.loads(raw)
    except Exception:
        print(json.dumps({}))
        return

    status = payload.get("status")
    loop_count = int(payload.get("loop_count", 0))

    # Safety: only loop if explicitly enabled via flag file
    enable_flag = Path(".cursor/state/ralf/ENABLE_STOP_HOOK")
    if not enable_flag.exists():
        print(json.dumps({}))
        return

    if status != "completed" or loop_count >= MAX_ITERATIONS:
        print(json.dumps({}))
        return

    tasks_path = Path(".cursor/state/gsd/TASKS.yaml")
    if not tasks_path.exists():
        print(json.dumps({}))
        return

    # Lightweight heuristic: check if any tasks still need work
    tasks_text = tasks_path.read_text(encoding="utf-8", errors="ignore")
    remaining = (
        ("status: todo" in tasks_text)
        or ("status: blocked" in tasks_text)
        or ("status: doing" in tasks_text)
    )

    if remaining:
        msg = (
            f"[RALF Stop Hook {loop_count + 1}/{MAX_ITERATIONS}] "
            f"Remaining tasks detected. Continue with `/ralf`."
        )
        print(json.dumps({"followup_message": msg}))
    else:
        print(json.dumps({}))


if __name__ == "__main__":
    main()

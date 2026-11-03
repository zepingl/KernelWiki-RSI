"""Run one claude call and keep its tool events, so a run can tell which knowledge files an agent opened.

Usage (as the runtime's agent binary): python3 agent_events.py <claude args...>, prompt on stdin.
The real CLI is RSI_CLAUDE_BIN. With RSI_EVENTS_DIR set, the call runs with stream-json output, every event line
goes to RSI_EVENTS_DIR/<id>.jsonl (first line names the lane phase), and stdout carries only the final result
text, exactly what the runtime receives from plain text mode. Without RSI_EVENTS_DIR it is a transparent exec.
"""

import json
import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path


def main(argv):
    claude = os.environ.get("RSI_CLAUDE_BIN")
    if not claude:
        print("agent_events: RSI_CLAUDE_BIN is not set", file=sys.stderr)
        return 2
    events_dir = os.environ.get("RSI_EVENTS_DIR")
    if not events_dir:
        os.execv(claude, [claude, *argv])
    if "--output-format" in argv:
        print("agent_events: the caller must not choose the output format", file=sys.stderr)
        return 2
    prompt = sys.stdin.read()
    phase = re.search(r"PHASE=(\w+)", prompt)
    path = Path(events_dir) / f"{time.strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen([claude, *argv, "--output-format", "stream-json", "--verbose"],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    process.stdin.write(prompt)
    process.stdin.close()
    result = None
    with path.open("x") as stream:
        stream.write(json.dumps({"agent_events": 1, "phase": phase[1] if phase else None}) + "\n")
        for line in process.stdout:
            stream.write(line if line.endswith("\n") else line + "\n")
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if isinstance(event, dict) and event.get("type") == "result":
                result = event
    code = process.wait()
    if result is None or result.get("is_error"):
        print(f"agent_events: no successful result event (exit {code}); events kept in {path}", file=sys.stderr)
        return code or 1
    sys.stdout.write(str(result.get("result", "")))
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

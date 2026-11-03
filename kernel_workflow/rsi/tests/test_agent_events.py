import json
import subprocess
import sys
from pathlib import Path

WRAPPER = Path(__file__).resolve().parents[1] / "agent_events.py"


def fake(tmp_path, events):
    claude = tmp_path / "claude"
    claude.write_text("#!/usr/bin/python3\nimport json, sys\nsys.stdin.read()\n"
                      "if '--output-format' not in sys.argv:\n    print('plain'); sys.exit(0)\n"
                      f"for event in {events!r}:\n    print(json.dumps(event))\n")
    claude.chmod(0o755)
    return claude


def run(claude, environment, prompt="PHASE=plan_round\nhello"):
    return subprocess.run([sys.executable, str(WRAPPER), "-p"], input=prompt, capture_output=True, text=True,
                          env={"PATH": "/usr/bin:/bin", "RSI_CLAUDE_BIN": str(claude), **environment})


def test_without_an_events_dir_the_wrapper_is_transparent(tmp_path):
    assert run(fake(tmp_path, []), {}).stdout == "plain\n"


def test_events_are_kept_and_only_the_final_result_is_printed(tmp_path):
    events = [{"type": "assistant", "message": {"content": [{"type": "tool_use", "id": "t", "name": "Read",
                                                              "input": {"file_path": "/x"}}]}},
              {"type": "result", "subtype": "success", "is_error": False, "result": "{\"ok\": true}"}]
    process = run(fake(tmp_path, events), {"RSI_EVENTS_DIR": str(tmp_path / "events")})
    assert process.returncode == 0 and process.stdout == "{\"ok\": true}"
    [log] = (tmp_path / "events").glob("*.jsonl")
    lines = [json.loads(line) for line in log.read_text().splitlines()]
    assert lines[0] == {"agent_events": 1, "phase": "plan_round"} and lines[1:] == events


def test_a_call_without_a_successful_result_fails_and_keeps_its_events(tmp_path):
    process = run(fake(tmp_path, [{"type": "result", "is_error": True, "result": "boom"}]),
                  {"RSI_EVENTS_DIR": str(tmp_path / "events")})
    assert process.returncode != 0 and process.stdout == "" and "no successful result" in process.stderr
    assert len(list((tmp_path / "events").glob("*.jsonl"))) == 1

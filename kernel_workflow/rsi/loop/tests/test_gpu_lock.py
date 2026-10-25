import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

GPU_LOCK = Path(__file__).resolve().parents[3] / "scripts/gpu_lock.sh"


@pytest.fixture
def wrapper_environment(tmp_path):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    probe = binaries / "rocminfo"
    probe.write_text(
        '#!/bin/bash\nprintf "%s|%s" "${HIP_VISIBLE_DEVICES:-}" "${ROCR_VISIBLE_DEVICES:-}" > "$PROBE_LOG"\n'
        'printf "  Name: gfx950\\n"\n'
    )
    probe.chmod(0o755)
    reaper = binaries / "pgrep"
    reaper.write_text('#!/bin/bash\nprintf invoked > "$REAPER_LOG"\nexit 1\n')
    reaper.chmod(0o755)
    environment = {key: os.environ[key] for key in ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "PYTHONPATH")
                   if key in os.environ}
    environment.update({
        "PATH": str(binaries) + os.pathsep + os.environ["PATH"],
        "KERNEL_EXTRACT_LOCK_DIR": str(tmp_path / "locks"),
        "TORCH_EXTENSIONS_DIR": str(tmp_path / "extensions"),
        "GEAK_GPU_REQUIRE_IDLE": "0",
        "PROBE_LOG": str(tmp_path / "probe.txt"),
        "REAPER_LOG": str(tmp_path / "reaper.txt"),
        "NRC_RUN_ID": "nrc-fixture", "NRC_NODE": "fixture-node", "NRC_CPUSET": "0",
        "NRC_CPU_CORES": "1", "NRC_LOGICAL_GPU_IDS": "0,1",
        "NRC_PHYSICAL_GPU_PCI": "0000:ab:00.0,0000:cd:00.0", "HIP_VISIBLE_DEVICES": "6,3",
    })
    return environment


def run_wrapper(tmp_path, environment, gpu_spec):
    command = (
        "import json, os; from pathlib import Path; "
        "Path('executed.json').write_text(json.dumps({key: os.environ.get(key) for key in "
        "['HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES', 'GEAK_GPU_ID']}))"
    )
    return subprocess.run(["bash", str(GPU_LOCK), gpu_spec, sys.executable, "-c", command],
                          cwd=tmp_path, env=environment, text=True, capture_output=True, timeout=15, check=False)


@pytest.mark.parametrize("gpu_spec,selected", [("0", "0"), ("1", "1"), ("0,1", "0")])
def test_coordinated_wrapper_preserves_visibility_and_selects_a_logical_device(tmp_path, wrapper_environment, gpu_spec, selected):
    result = run_wrapper(tmp_path, wrapper_environment, gpu_spec)
    assert result.returncode == 0, result.stderr
    assert json.loads((tmp_path / "executed.json").read_text()) == {
        "HIP_VISIBLE_DEVICES": "6,3", "ROCR_VISIBLE_DEVICES": None,
        "CUDA_VISIBLE_DEVICES": None, "GEAK_GPU_ID": selected,
    }
    assert (tmp_path / "probe.txt").read_text() == "6,3|"
    assert not (tmp_path / "reaper.txt").exists()


@pytest.mark.parametrize("gpu_spec,overrides", [
    ("2", {}), ("0,,1", {}), ("0", {"ROCR_VISIBLE_DEVICES": "0"}),
    ("0", {"CUDA_VISIBLE_DEVICES": "0"}), ("0", {"HIP_VISIBLE_DEVICES": ""}),
    ("0", {"NRC_PHYSICAL_GPU_PCI": "0000:ab:00.0"}),
    ("0", {"GEAK_GPU_ALLOWED": "1"}),
])
def test_coordinated_wrapper_rejects_invalid_placement_before_command(tmp_path, wrapper_environment, gpu_spec, overrides):
    wrapper_environment.update(overrides)
    result = run_wrapper(tmp_path, wrapper_environment, gpu_spec)
    assert result.returncode != 0
    assert not (tmp_path / "executed.json").exists()


def test_uncoordinated_wrapper_keeps_legacy_device_selection(tmp_path, wrapper_environment):
    environment = {key: value for key, value in wrapper_environment.items() if not key.startswith("NRC_")}
    environment["KERNEL_ENV_SKIP_ENUM_REAP"] = "1"
    result = run_wrapper(tmp_path, environment, "7")
    assert result.returncode == 0, result.stderr
    assert json.loads((tmp_path / "executed.json").read_text())["HIP_VISIBLE_DEVICES"] == "7"
    assert (tmp_path / "probe.txt").read_text() == "7|7"


def test_coordinated_wrapper_does_not_measure_an_unknown_physical_device(tmp_path, wrapper_environment):
    wrapper_environment.update({"GEAK_GPU_REQUIRE_IDLE": "1",
                                "NRC_PHYSICAL_GPU_PCI": "ffff:ff:1f.7,ffff:ff:1f.6"})
    result = run_wrapper(tmp_path, wrapper_environment, "0")
    assert result.returncode != 0
    assert not (tmp_path / "executed.json").exists()

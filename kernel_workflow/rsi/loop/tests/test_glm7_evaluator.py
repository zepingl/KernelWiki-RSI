import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from kernel_workflow.rsi import glm7_evaluator
from kernelwiki.knowledge import KnowledgeError


def test_adapter_preserves_original_ut_measurements_and_adds_only_device_identity(tmp_path, monkeypatch):
    report = {"schema_version": 2, "runtime_fingerprint": "fixture-runtime", "correctness_pass": True,
              "per_case": [{"name": "original", "baseline_ms": 2, "optimized_ms": 1}]}
    source = "from pathlib import Path\nPath('ut_result.json').write_text(" + repr(json.dumps(report)) + ")\n"
    (tmp_path / "unittest.py").write_text(source)
    calls = []
    identity = {"node": "fixture-node", "pci": "0000:01:00.0"}
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(
        set_device=lambda device: calls.append(device), current_device=lambda: 0)))
    monkeypatch.setattr(glm7_evaluator, "device_identity", lambda device: identity)
    monkeypatch.setattr(sys, "argv", ["fixture"])
    original_path = list(sys.path)
    assert glm7_evaluator.main(["--gpu-id", "0"]) == 0
    assert calls == [0]
    assert json.loads((tmp_path / "ut_result.json").read_text()) == {**report, "device_identity": identity}
    assert (tmp_path / "unittest.py").read_text() == source
    assert sys.argv == ["fixture"]
    assert sys.path == original_path


def test_adapter_cannot_reuse_an_old_report_after_failed_ut(tmp_path, monkeypatch):
    (tmp_path / "unittest.py").write_text("raise SystemExit(1)\n")
    (tmp_path / "ut_result.json").write_text('{"correctness_pass": true}')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(set_device=lambda device: None)))
    monkeypatch.setattr(glm7_evaluator, "device_identity", lambda device: {"node": "fixture", "pci": "fixture"})
    monkeypatch.setattr(sys, "argv", ["fixture"])
    with pytest.raises(SystemExit) as stopped:
        glm7_evaluator.main(["--gpu-id", "0"])
    assert stopped.value.code == 1
    assert not (tmp_path / "ut_result.json").exists()


def test_adapter_rejects_unqualified_legacy_report(tmp_path, monkeypatch):
    (tmp_path / "unittest.py").write_text("from pathlib import Path\nPath('ut_result.json').write_text('{}')\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(
        set_device=lambda device: None, current_device=lambda: 0)))
    monkeypatch.setattr(glm7_evaluator, "device_identity", lambda device: {"node": "fixture", "pci": "fixture"})
    monkeypatch.setattr(sys, "argv", ["fixture"])
    with pytest.raises(KnowledgeError, match="original schema-2"):
        glm7_evaluator.main(["--gpu-id", "0"])


def test_adapter_rejects_a_changed_runtime_device(tmp_path, monkeypatch):
    (tmp_path / "unittest.py").write_text("pass\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(
        set_device=lambda device: None, current_device=lambda: 1)))
    monkeypatch.setattr(glm7_evaluator, "device_identity", lambda device: {"node": "fixture", "pci": "0000:01:00.0"})
    monkeypatch.setattr(sys, "argv", ["fixture"])
    with pytest.raises(KnowledgeError, match="changed the selected runtime device"):
        glm7_evaluator.main(["--gpu-id", "0"])


@pytest.mark.parametrize("bus_address,exit_code,node,expected", [
    (b"00000000:AB:00.0", 0, "fixture-node", "0000:ab:00.0"),
    (b"not-a-pci-address", 0, "fixture-node", None),
    (b"0000:ab:00.0", 0, None, None),
    (b"0000:ab:00.0", 1, "fixture-node", None),
])
def test_device_identity_uses_the_observed_hip_pci(bus_address, exit_code, node, expected, monkeypatch):
    def query(bus, size, device):
        assert size == 64
        assert device == 0
        bus.value = bus_address
        return exit_code

    monkeypatch.setattr(glm7_evaluator.ctypes, "CDLL", lambda path: SimpleNamespace(hipDeviceGetPCIBusId=query))
    monkeypatch.setattr(glm7_evaluator.ctypes.util, "find_library", lambda name: "/fixture/libamdhip64.so")
    if node:
        monkeypatch.setenv("NRC_NODE", node)
    else:
        monkeypatch.delenv("NRC_NODE", raising=False)
    if expected:
        assert glm7_evaluator.device_identity(0) == {"node": node, "pci": expected}
    else:
        with pytest.raises(KnowledgeError, match="HIP"):
            glm7_evaluator.device_identity(0)


def test_adapter_imports_standard_unittest_instead_of_the_local_ut(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    report = {"schema_version": 2, "runtime_fingerprint": "fixture-runtime", "correctness_pass": True}
    (workspace / "unittest.py").write_text(
        "from pathlib import Path\nPath('ut_result.json').write_text(" + repr(json.dumps(report)) + ")\n")
    dependencies = tmp_path / "dependencies"
    dependencies.mkdir()
    (dependencies / "torch.py").write_text(
        "import unittest\nfrom types import SimpleNamespace\n"
        "assert hasattr(unittest, 'TestCase'), 'workspace unittest shadowed the standard library'\n"
        "cuda = SimpleNamespace(set_device=lambda device: None, current_device=lambda: 0)\n")
    roots = {str(Path(glm7_evaluator.__file__).resolve().parents[2]), str(dependencies)}
    roots.update(str(Path(path).resolve()) for path in sys.path if path)
    environment = {key: os.environ[key] for key in ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR")
                   if key in os.environ}
    environment["PYTHONPATH"] = os.pathsep.join(sorted(roots))
    command = ("from kernel_workflow.rsi import glm7_evaluator; "
               "glm7_evaluator.device_identity = lambda device: {'node': 'fixture', 'pci': '0000:01:00.0'}; "
               "glm7_evaluator.main(['--gpu-id', '0'])")
    result = subprocess.run([sys.executable, "-c", command], cwd=workspace, env=environment,
                            text=True, capture_output=True, timeout=15, check=False)
    assert result.returncode == 0, result.stderr
    assert json.loads((workspace / "ut_result.json").read_text())["runtime_fingerprint"] == "fixture-runtime"

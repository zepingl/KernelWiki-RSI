"""Run an unchanged standalone GLM7 UT and bind its report to the actual HIP device."""

import argparse
import ctypes
import ctypes.util
import os
import re
import runpy
import sys
from pathlib import Path

from kernelwiki._atomic import atomic_bytes
from kernelwiki.knowledge import KnowledgeError, canonical

from .evidence import read_json


def device_identity(gpu_id):
    library = ctypes.CDLL(ctypes.util.find_library("amdhip64") or "/opt/rocm/lib/libamdhip64.so")
    library.hipDeviceGetPCIBusId.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    library.hipDeviceGetPCIBusId.restype = ctypes.c_int
    bus = ctypes.create_string_buffer(64)
    code = library.hipDeviceGetPCIBusId(bus, len(bus), gpu_id)
    if code:
        raise KnowledgeError(f"HIP physical device query failed: {code}")
    match = re.fullmatch(r"([0-9a-fA-F]{4,8}):([0-9a-fA-F]{2}):([0-9a-fA-F]{2}\.[0-7])", bus.value.decode())
    if not match or not os.environ.get("NRC_NODE"):
        raise KnowledgeError("GLM7 evaluation requires observed HIP PCI identity and an NRC placement")
    domain, bus_number, device = match.groups()
    return {"node": os.environ["NRC_NODE"], "pci": f"{int(domain, 16):04x}:{bus_number.lower()}:{device.lower()}"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-id", type=int, required=True)
    args = parser.parse_args(argv)
    workspace = Path.cwd().resolve()
    entry = Path("unittest.py").resolve(strict=True)
    report_path = Path("ut_result.json")
    if report_path.is_symlink():
        raise KnowledgeError("GLM7 report cannot be a symlink")
    report_path.unlink(missing_ok=True)
    original_path, original_argv = list(sys.path), sys.argv
    try:
        sys.path[:] = [path for path in original_path if Path(path or ".").resolve() != workspace]
        import torch
        torch.cuda.set_device(args.gpu_id)
        identity = device_identity(args.gpu_id)
        sys.argv = [str(entry)]
        try:
            runpy.run_path(str(entry), run_name="__main__")
        except SystemExit as stopped:
            if stopped.code not in (0, None):
                raise
        if torch.cuda.current_device() != args.gpu_id or device_identity(args.gpu_id) != identity:
            raise KnowledgeError("GLM7 evaluation changed the selected runtime device")
        report = read_json(report_path)
        if report.get("schema_version") != 2 or not report.get("runtime_fingerprint"):
            raise KnowledgeError("GLM7 evaluation needs the original schema-2 runtime-bound report")
        report["device_identity"] = identity
        atomic_bytes(report_path, canonical(report), durable=True)
        return 0
    finally:
        sys.path[:] = original_path
        sys.argv = original_argv


if __name__ == "__main__":
    raise SystemExit(main())

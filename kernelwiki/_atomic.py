"""Atomic local file replacement shared by the two storage contracts."""

import os
import tempfile


def atomic_bytes(path, payload, *, durable=False):
    path = os.fspath(path)
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="." + os.path.basename(path) + ".", dir=directory)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = ""
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)
    fsync_dir(directory, strict=durable)


def fsync_dir(path, *, strict=False):
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        if strict:
            raise
        return
    try:
        os.fsync(descriptor)
    except OSError:
        if strict:
            raise
    finally:
        os.close(descriptor)

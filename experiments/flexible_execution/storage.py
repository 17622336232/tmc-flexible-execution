"""Losslessly archive simulator diagnostics after their writers have stopped."""

import gzip
from hashlib import sha256
import json
import os

from experiments.task_attack.runtime import write_json


def archive_sensor_logs(root):
    private = root / "private"
    paths = (private / "sensor_timing.jsonl", *sorted((private / "roslog").rglob("*.log")))
    manifest = root / "storage_archive.json"
    stored = []
    try:
        saved = json.loads(manifest.read_text(encoding="utf8")) if manifest.is_file() else {"diagnostic_logs": []}
        entries = {row["original"]: row for row in saved["diagnostic_logs"]}
        for source in paths:
            if not source.is_file() or source.is_symlink():
                continue
            raw = source.read_bytes()
            target = source.with_suffix(source.suffix + ".gz")
            if target.is_file():
                verified = target
            else:
                verified = target.with_suffix(".gz.tmp")
                with verified.open("wb") as handle:
                    handle.write(gzip.compress(raw, compresslevel=6, mtime=0))
                    handle.flush()
                    os.fsync(handle.fileno())
            if gzip.decompress(verified.read_bytes()) != raw:
                raise ValueError("The diagnostic archive failed its round-trip check")
            if verified != target:
                verified.replace(target)
            digest = sha256(raw).hexdigest()
            name = str(source.relative_to(root))
            entries[name] = {"original": name, "gzip": str(target.relative_to(root)), "sha256": digest,
                             "original_bytes": len(raw), "compressed_bytes": target.stat().st_size}
            stored.append((source, digest))
        if stored:
            write_json(manifest, {"format": "gzip", "diagnostic_logs": list(entries.values())})
        for source, digest in stored:
            if sha256(source.read_bytes()).hexdigest() == digest:
                source.unlink()
    except (OSError, ValueError, EOFError) as error:
        write_json(root / "storage_archive_error.json", {"type": type(error).__name__, "message": str(error)})

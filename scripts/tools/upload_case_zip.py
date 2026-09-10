#!/usr/bin/env python3
"""Create a Workbench case and upload a DICOM zip."""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path


def _multipart(path: Path) -> tuple[bytes, str]:
    boundary = "----wbuploadboundary"
    name = path.name
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{name}"\r\n'
        "Content-Type: application/zip\r\n\r\n"
    ).encode("utf-8") + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode("ascii")
    return body, f"multipart/form-data; boundary={boundary}"


def main() -> int:
    api = sys.argv[1].rstrip("/")
    zip_path = Path(sys.argv[2])
    label = sys.argv[3] if len(sys.argv) > 3 else zip_path.stem
    if not zip_path.exists():
        raise SystemExit(f"zip not found: {zip_path}")
    create = urllib.request.Request(
        f"{api}/api/v1/cases",
        data=json.dumps({"patient_label": label}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(create, timeout=30) as resp:
        meta = json.loads(resp.read().decode("utf-8"))
    case_id = meta["case_id"]
    body, content_type = _multipart(zip_path)
    upload = urllib.request.Request(
        f"{api}/api/v1/cases/{case_id}/upload",
        data=body,
        headers={"Content-Type": content_type},
        method="POST",
    )
    with urllib.request.urlopen(upload, timeout=600) as resp:
        up = json.loads(resp.read().decode("utf-8"))
    print(json.dumps({"case_id": case_id, "upload": up}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

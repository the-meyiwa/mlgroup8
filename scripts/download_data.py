#!/usr/bin/env python3
"""Fetch the training corpus.

Tiny Shakespeare is the standard character-level benchmark: 1.1MB of public domain plays,
40 characters of vocabulary, small enough to overfit in seconds and large enough to show
whether a model is learning anything.
"""

import sys
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DEST = Path(__file__).resolve().parent.parent / "data" / "tinyshakespeare.txt"


def main() -> int:
    DEST.parent.mkdir(parents=True, exist_ok=True)

    if DEST.exists() and DEST.stat().st_size > 100_000:
        print(f"already have {DEST} ({DEST.stat().st_size} bytes)")
        return 0

    print(f"downloading {URL}")
    try:
        with urllib.request.urlopen(URL, timeout=30) as r:
            data = r.read()
    except Exception as exc:  # noqa: BLE001 - report and fall back
        print(f"download failed: {exc}", file=sys.stderr)
        print(f"put any plain text file at {DEST} and rerun", file=sys.stderr)
        return 1

    DEST.write_bytes(data)
    print(f"wrote {DEST} ({len(data)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

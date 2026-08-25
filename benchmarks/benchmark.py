"""Small end-to-end CPU benchmark that writes to the actual filesystem."""

from __future__ import annotations

import argparse
import hashlib
import tempfile
import time
from pathlib import Path

import generateCombinations as generator


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=6)
    parser.add_argument("--count", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=generator.DEFAULT_BATCH_SIZE)
    args = parser.parse_args()

    limit = min(args.count, 10**args.length)
    with tempfile.TemporaryDirectory(prefix="corestorm-benchmark-") as directory:
        output = Path(directory) / "combinations.txt"
        started = time.perf_counter()
        exit_code = generator.main(
            [
                str(args.length),
                "--stop",
                str(limit),
                "--output",
                str(output),
                "--batch-size",
                str(args.batch_size),
                "--force",
                "--yes",
                "--progress-interval",
                "0",
            ]
        )
        elapsed = time.perf_counter() - started
        digest_builder = hashlib.sha256()
        with output.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest_builder.update(chunk)
        digest = digest_builder.hexdigest()
        print(f"exit_code={exit_code}")
        print(f"combinations={limit}")
        print(f"bytes={output.stat().st_size}")
        print(f"seconds={elapsed:.4f}")
        print(f"combinations_per_second={limit / elapsed:.0f}")
        print(f"sha256={digest}")
        return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

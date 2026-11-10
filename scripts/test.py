#!/usr/bin/env python3
"""Run each test file in its own process, isolating score-generator imports."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    files = [Path(p) for p in sys.argv[1:]] or sorted((ROOT / "tests").glob("test_*.py"))
    failed = []
    for path in files:
        print(f"\n=== {path.name} ===", flush=True)
        result = subprocess.run([sys.executable, "-m", "pytest", "-q", "-rs", str(path)], cwd=ROOT)
        if result.returncode:
            failed.append(path.name)
    print(f"\n{len(files) - len(failed)}/{len(files)} test files passed", flush=True)
    if failed:
        print("Failed: " + ", ".join(failed), file=sys.stderr)
    return bool(failed)


if __name__ == "__main__":
    sys.exit(main())

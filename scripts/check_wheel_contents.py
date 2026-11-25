# SPDX-License-Identifier: Apache-2.0
"""Validate that the wheel contains only the intended LMCache subtree."""

# Standard
from pathlib import Path
from zipfile import ZipFile
import sys


def main() -> None:
    """Check wheel namespace ownership and required package data."""
    wheel = Path(sys.argv[1])
    with ZipFile(wheel) as archive:
        names = set(archive.namelist())

    package_files = {name for name in names if name.startswith("lmcache/")}
    compatibility_files = {
        "lmcache/integration/vllm/vllm_v1_adapter.py",
        "lmcache/integration/tensorrt_llm/tensorrt_adapter.py",
    }
    unexpected = sorted(
        name
        for name in package_files
        if not name.startswith("lmcache/inprocess/") and name not in compatibility_files
    )
    if unexpected:
        raise SystemExit(f"wheel owns shared LMCache files: {unexpected}")

    required = {
        "lmcache/inprocess/__init__.py",
        "lmcache/inprocess/v1/cache_engine.py",
        "lmcache/inprocess/v1/cache_controller/frontend/static/index.html",
        "lmcache/integration/vllm/vllm_v1_adapter.py",
        "lmcache/integration/tensorrt_llm/tensorrt_adapter.py",
    }
    missing = sorted(required - names)
    if missing:
        raise SystemExit(f"wheel is missing required files: {missing}")

    print(f"validated {wheel}: {len(package_files)} in-process package files")


if __name__ == "__main__":
    main()

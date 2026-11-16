#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
revision_file="${repo_root}/UPSTREAM_REVISION"

if [[ ! -s "${revision_file}" ]]; then
  echo "missing upstream revision file: ${revision_file}" >&2
  exit 1
fi

upstream_revision="$(sed -n 's/^revision=//p' "${revision_file}" | head -n 1)"
if [[ ! "${upstream_revision}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "UPSTREAM_REVISION must start with a 40-character commit SHA" >&2
  exit 1
fi

if command -v python >/dev/null 2>&1; then
  python_bin="$(command -v python)"
elif command -v python3.12 >/dev/null 2>&1; then
  python_bin="$(command -v python3.12)"
else
  python_bin="$(command -v python3)"
fi

if command -v uv >/dev/null 2>&1; then
  installer=(uv pip install)
else
  installer=("${python_bin}" -m pip install)
fi

echo "Installing LMCache dependency at ${upstream_revision}"
"${installer[@]}" \
  "lmcache @ git+https://github.com/LMCache/LMCache.git@${upstream_revision}"

echo "Installing the current lmcache_inprocess checkout"
# A regular wheel overwrites the legacy inprocess tree bundled by older LMCache
# versions; a PEP 660 editable finder loses to that existing package on sys.path.
"${installer[@]}" --no-deps --force-reinstall "${repo_root}"

"${python_bin}" - <<'PY'
import lmcache
import lmcache.inprocess

print(f"LMCache dependency: {lmcache.__file__}")
print(f"LMCache InProcess checkout: {lmcache.inprocess.__file__}")
PY

# LMCache InProcess

`lmcache_inprocess` packages LMCache's legacy in-process engine separately from
the main [LMCache](https://github.com/LMCache/LMCache) distribution. It keeps the
existing Python import path, `lmcache.inprocess`, so current vLLM, TensorRT-LLM,
configuration, and plugin integrations do not need an API rename.

The wheel also supplies
`lmcache.integration.vllm.vllm_v1_adapter` as a compatibility re-export because
vLLM's connector imports that canonical path. The implementation remains under
`lmcache.inprocess.integration.vllm`.

The source snapshot is synchronized through LMCache `dev` commit
[`05fdc2013b41484f0d834ec2d30b019fd426a56c`](https://github.com/LMCache/LMCache/commit/05fdc2013b41484f0d834ec2d30b019fd426a56c).
The exact source revision is also recorded in [`UPSTREAM_REVISION`](UPSTREAM_REVISION).

## Install

Install a compatible LMCache build first, then this package:

```bash
python -m pip install 'lmcache>=0.5.7rc3'
python -m pip install lmcache_inprocess
```

For development against LMCache `dev`:

```bash
NO_GPU_EXT=1 python -m pip install \
  'lmcache @ git+https://github.com/LMCache/LMCache.git@dev'
python -m pip install -e . --no-deps
```

The distribution is pure Python. Native extensions remain owned and shipped by
the `lmcache` dependency. A CLI-only `NO_NATIVE_EXT=1` LMCache installation is
not sufficient because the in-process engine requires LMCache's common native
operations; CPU validation uses `NO_GPU_EXT=1`.

## Compatibility contract

- Distribution name: `lmcache_inprocess`
- Import namespace: `lmcache.inprocess`
- Compatibility imports: `lmcache.integration.vllm.vllm_v1_adapter` and
  `lmcache.integration.tensorrt_llm.tensorrt_adapter`
- Base dependency: `lmcache>=0.5.7rc3`
- Python: 3.10 through 3.14
- Supported source of shared native extensions: the installed `lmcache` package

Because both distributions contribute files below the top-level `lmcache`
directory, install and uninstall them with the same environment manager. A
compatible LMCache release must stop shipping each extracted path, including
the compatibility imports above, to avoid overlapping wheel ownership.

TensorRT-LLM's `lmcache` connector preset imports the compatibility path above,
which forwards to `lmcache.inprocess.integration.tensorrt_llm.tensorrt_adapter`.
Install this distribution for that preset. The `lmcache-mp` adapter remains
owned by LMCache. When upgrading across the extraction, upgrade LMCache first
and then install or reinstall this distribution so the shim remains installed.

## Development and validation

```bash
python -m pip install -r requirements/test.txt
python -m build
python -m twine check dist/*
python -m pytest -q tests/smoke
python -m pytest -q tests/inprocess/v1/test_mock_gpu_connector.py
```

CI builds and installs the wheel in a clean environment, runs CPU smoke tests,
and tests compatibility against a pinned LMCache revision and current LMCache
`dev`. GPU end-to-end validation is available through the manual workflow in
`.github/workflows/gpu-e2e.yml` on a repository runner labeled
`self-hosted`, `linux`, `x64`, `gpu`, and `h200`.

## Repository layout

- `lmcache/inprocess/`: package source
- `lmcache/integration/`: compatibility shims for inference-engine import paths
- `tests/inprocess/`: copied unit and integration tests
- `examples/inprocess/`: in-process examples
- `benchmarks/inprocess/`: in-process benchmarks
- `docs/source/legacy/`: in-process documentation migrated from LMCache
- `scripts/sync_from_lmcache.sh`: reproducible upstream refresh helper

## Release

Create a GitHub release with a `vX.Y.Z` tag. The release workflow builds the
source distribution and universal wheel, validates both, uploads them as GitHub
release artifacts, and publishes to PyPI through Trusted Publishing when the
`pypi` environment is configured for this repository.

---
layout: default
title: Development
---

# Development

Jackdaw is a mixed Python/Rust project. The Python package lives in `jackdaw/`, and the Rust
workspace in `crates/` builds the native module `jackdaw._core` with [PyO3] and [maturin].
Pull requests go to the `dev` branch; [CONTRIBUTING.md](https://github.com/TylerFlar/jackdaw-balatro/blob/main/CONTRIBUTING.md)
has the branch flow, the checklist and how a release is cut.

| Path | Role |
|---|---|
| `crates/jackdaw-data` | Generated data tables and enums (`cargo xtask codegen`) |
| `crates/jackdaw-core` | The game engine |
| `crates/jackdaw-bot` | BalatroBot command and gamestate layer |
| `crates/jackdaw-env` | Observation encoder, masks, vector env |
| `crates/jackdaw-py` | PyO3 bindings, built as `jackdaw._core` |
| `xtask` | Repository automation (`cargo xtask ...`) |

## Prerequisites

- [uv].
- Rust via [rustup]. `rust-toolchain.toml` pins the version (with rustfmt and clippy), and
  rustup installs it the first time you run `cargo` in the repo.
- On Windows, either toolchain works:
  - **GNU** is fine for development. It needs MinGW-w64 `gcc` and `dlltool` on `PATH`, e.g.
    from MSYS2 UCRT64 (`pacman -S mingw-w64-ucrt-x86_64-toolchain`). PyO3 links `python3.dll`
    with `raw-dylib`, which calls `dlltool` on this target. The toolchain file names no target
    triple, so make GNU rustup's default host:
    `rustup set default-host x86_64-pc-windows-gnu`. Without it, rustup tries to install
    `1.98.1-x86_64-pc-windows-msvc`.
  - **MSVC** is recommended for release builds and performance work; CI builds the Windows
    wheels with it. Install the Visual Studio Build Tools ("Desktop development with C++") and
    keep rustup's default host at `x86_64-pc-windows-msvc`. If the Windows SDK is missing and
    you can't install it (no admin rights), `cargo install xwin` and
    `xwin --accept-license splat --output <dir>` put the CRT and SDK in a folder of your own.
    Point `INCLUDE` and `LIB` at its `crt` and `sdk` subfolders as well as MSVC's own, and set
    `VCINSTALLDIR` so the `cc` crate uses that environment as it is.

Installing jackdaw from a git URL or an sdist compiles the extension, so it needs Rust as well.
The prebuilt wheels on the GitHub releases do not.

## Everyday commands

```bash
uv sync --dev                        # build the extension, install jackdaw in editable mode
uv run pytest                        # Python tests (rebuilds first if the Rust side changed)
uv sync --reinstall-package jackdaw  # force a rebuild

cargo test                           # Rust tests; skips jackdaw-py, never links libpython
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo xtask help
```

`uv sync` builds the extension with maturin and installs jackdaw in editable mode. Python edits
take effect immediately. The compiled module is copied into the package as `jackdaw/_core.pyd`
(Windows) or `jackdaw/_core.abi3.so` (Linux, macOS); both are gitignored. `[tool.uv] cache-keys`
in `pyproject.toml` lists the Rust inputs (Cargo manifests, `Cargo.lock`, the toolchain file,
`crates/**/*.rs`), so `uv sync` and `uv run` rebuild by themselves after a Rust change.

Editable builds use the release profile (fat LTO). For quicker, unoptimized rebuilds while
iterating on Rust code, pass a profile to maturin:

```bash
MATURIN_PEP517_ARGS="--profile dev" uv sync --reinstall-package jackdaw
# PowerShell: $env:MATURIN_PEP517_ARGS = "--profile dev"; uv sync --reinstall-package jackdaw
```

The variable applies to every rebuild in that shell. Unset it and run
`uv sync --reinstall-package jackdaw` to go back to release. To see what is installed, run
`uv run python -c "import jackdaw._core as c; print(c._build_info())"`.

`uv build` writes an sdist and an abi3 wheel (CPython 3.12+) to `dist/`. The `Wheels` workflow
builds the release wheels for Linux, macOS and Windows on `v*` tags.

## Game data and `cargo xtask`

`cargo xtask` is a small CLI with no argument-parsing dependency:

- `codegen [--check]` generates `crates/jackdaw-data/src/generated.rs` from
  `jackdaw/engine/data/*.json`. CI runs it with `--check`, which fails if the committed file is
  stale.
- `verify-source` checks that JSON against the game itself. It runs
  `scripts/extract_prototypes.lua` and `scripts/extract_game_tables.lua` on `balatro_source/`
  into a temporary directory and compares each file with the committed one value by value:
  formatting and object key order do not matter, array order does, and numbers must be the same
  double. It lists the JSON path of every difference and exits non-zero if there is any. To
  update the committed data instead, run the two scripts without an output directory, then
  `cargo xtask codegen`.
- `lua-oracles` reruns the LuaJIT oracles behind the `tests/fixtures/*_luajit*.json` fixtures
  and rewrites the hands, challenges and blind-amount JSON. `rng_luajit.json` holds exact RNG
  values; `rng_luajit_bulk.json` holds one digest per run seed of about 10^6 draws, which
  `crates/jackdaw-core/tests/rng_luajit_bulk.rs` replays. When a run differs, its LuaJIT values
  come from `luajit scripts/luajit_rng_bulk_oracle.lua --trace N`.

`verify-source` and `lua-oracles` need LuaJIT 2.1 on `PATH` and the game's Lua source in
`balatro_source/` at the repository root (gitignored: the source is proprietary).
`verify-source` also requires the first line of its `version.jkr` to be the release the data
targets, `BALATRO_VERSION` in `crates/jackdaw-data` (`1.0.1o-FULL`). CI has no copy of the
source, so these two commands run only locally.

## Frozen fixtures

The Rust engine was checked step by step against a Python reference engine until that engine
was retired. Its verdicts are frozen in `tests/fixtures`: golden v2's digest chains along the
recorded tapes (`parity_specs.json`, `golden_v2.json`), each step's digest for the joker
scenarios, fuzz cases and regressions, the new runs and setups, the env layer's observations,
masks and conversions, the environments' runs, and the targeted parity tests' steps
(`parity_cases.json`). `tests/parity/frozen.py` lists them.

A deliberate behaviour change makes the engine disagree with them. Once the change is
reviewed, rewrite the affected fixtures from the Rust engine, and say in the PR which ones
changed and why:

```bash
uv run python scripts/refreeze.py                    # every fixture
uv run python scripts/refreeze.py golden_v2 cases    # some of them
```

It replays every tape as recorded and recomputes its digests; a step the engine now refuses
ends that tape there.

## Gap numbers in comments

Comments and tests cite gap numbers such as `G206` or `G13`. They are rows of the gap register
the port kept while it audited the engine against the Lua source (G for gameplay, B for the
rest; not the deviation catalog's `B01`-`B04`). Every gap was fixed, and the register was
removed in 0.2.0; it is in the repository's history:

```bash
git show d176884:docs/parity/gaps.md
```

## Timings

`cargo test --release -p jackdaw-core --test perf -- --ignored --nocapture` prints the
engine's main operations (a new run, a copy, plays, the shop, an RNG draw), timed as an env runs
them, and `uv run pytest -m benchmark` the environments' steps per second. For a profiler
(`perf`, `samply`, VTune), `cargo build --profile profiling` builds release code with full
debug info.

## Versions

`pyproject.toml` and `Cargo.toml`'s `[workspace.package]` both hold the version, and
CHANGELOG.md needs a section for it; `uv run python scripts/check_version.py` checks that they
agree (CI runs it, and the Wheels workflow checks a release tag against it).

## Windows: the `.pyd` is locked while Python has it loaded

Windows does not let a loaded DLL be overwritten. If any Python process has imported
`jackdaw._core` (a REPL, a Jupyter kernel, a training run, a debugger, an IDE test runner), a
rebuild fails with:

```
failed to copy file from ...\target\maturin\_core.dll to ...\jackdaw\_core.pyd:
The process cannot access the file because it is being used by another process. (os error 32)
```

Stop those processes (restart the kernel) and run `uv sync` again.

[PyO3]: https://pyo3.rs
[maturin]: https://www.maturin.rs
[uv]: https://docs.astral.sh/uv/
[rustup]: https://rustup.rs

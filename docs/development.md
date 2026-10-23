# Development

Jackdaw is a mixed Python/Rust project. The Python package lives in `jackdaw/`, and the Rust
workspace in `crates/` builds the native module `jackdaw._core` with [PyO3] and [maturin].

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
    keep rustup's default host at `x86_64-pc-windows-msvc`.

Installing jackdaw from a git URL or an sdist compiles the extension, so it needs Rust as well.
Prebuilt wheels do not.

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

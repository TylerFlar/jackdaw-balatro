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

`cargo xtask` is a small std-only CLI. `codegen [--check]`, `verify-source` and `lua-oracles`
arrive in phase 2; until then they exit with status 2. `uv build` writes an sdist and an abi3
wheel (CPython 3.12+) to `dist/`. The `Wheels` workflow builds the release wheels for Linux,
macOS and Windows on `v*` tags.

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

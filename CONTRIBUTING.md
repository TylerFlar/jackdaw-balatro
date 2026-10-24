# Contributing

Thanks for helping. This page covers the branches, what a pull request needs, and how a release
is cut. [docs/development.md](docs/development.md) covers the toolchain and the everyday
commands.

## Branches

| branch | holds | pull requests from |
|---|---|---|
| `main` | the releases: each release is a merge from `dev`, tagged `vX.Y.Z` | `dev` (a release) or `hotfix/*` |
| `dev` | the next release | your branch |

**Open pull requests against `dev`**, the repository's default branch: branch from `dev`, and a
new pull request targets it. A required check (`pr-target`) fails any pull request into `main`
from another branch; if it fails on yours, change the base to `dev` (Edit, next to the title).

`main` only moves at a release, so installing from it gives the latest release
(`uv add git+https://github.com/TylerFlar/jackdaw-balatro.git --branch main`, or `--tag vX.Y.Z`
for a given one). An install from the bare URL gets `dev`.

## A pull request

CI runs on every pull request: ruff, rustfmt, clippy and the codegen check; the Rust tests on
Linux and Windows; the Python tests on 3.12, 3.13 and 3.14. Before you push:

```bash
uv run pytest
cargo test
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
uv run ruff check . && uv run ruff format --check .
```

- **Engine changes follow the game.** The engine reproduces Balatro 1.0.1o's Lua source; a fix
  comes with a regression test that cites the Lua lines (`card.lua:1234`). A deliberate change of
  behaviour makes the frozen fixtures disagree: refreeze them and say which ones changed and why
  ([development.md](docs/development.md#frozen-fixtures)).
- **Action-space changes** (a mask allows a move it didn't, a new action type) bump
  `ACTION_VERSION` in `crates/jackdaw-env/src/lib.rs`, since policies trained before no longer fit.
- **Changes users see** get a line under "Unreleased" in [CHANGELOG.md](CHANGELOG.md).
- Pull requests into `dev` can be squash-merged.

## Cutting a release

1. **On `dev`**, set the version:
   - rename CHANGELOG.md's "Unreleased" section to `## [X.Y.Z] - YYYY-MM-DD` and start a new,
     empty "Unreleased" above it;
   - set `version` in `pyproject.toml` and in `Cargo.toml`'s `[workspace.package]`;
   - refresh the lock files: `cargo update --workspace` and `uv lock`;
   - check: `uv run python scripts/check_version.py`.
2. **Open a pull request from `dev` to `main`** named "Release X.Y.Z", and merge it with a merge
   commit (not squash or rebase, so `main` and `dev` keep one history).
3. **Tag the merge commit** on `main` and push the tag:

   ```bash
   git switch main && git pull
   git tag -a vX.Y.Z -m "jackdaw X.Y.Z"
   git push origin vX.Y.Z
   ```

   The Wheels workflow checks the tag against the version, builds the wheels and the sdist, and
   opens a draft GitHub release with them and the changelog section. Publish it from the
   releases page.
4. **Bring `dev` up to `main`:** `git switch dev && git merge --ff-only main && git push`.

Versions follow [semantic versioning](https://semver.org). While the version is 0.x, a release
that breaks something (an action-version change, an API change) bumps the minor version
(0.2.0 → 0.3.0), and a release of fixes only bumps the patch version (0.2.0 → 0.2.1).

A fix that can't wait for the next release goes on a `hotfix/<name>` branch from `main`, with the
patch version bump and a changelog entry; open its pull request against `main`, tag it as above,
then merge `main` into `dev`. A fix to the docs or the workflows alone needs no version or tag.

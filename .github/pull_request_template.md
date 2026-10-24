<!--
Pull requests go to the `dev` branch: check the base branch above. Only a release (from `dev`)
or a `hotfix/*` branch targets `main`. See CONTRIBUTING.md.
-->

## What and why



## Checks

- [ ] `uv run pytest` and `cargo test` pass
- [ ] `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `ruff check .` and `ruff format --check .` are clean
- [ ] An engine fix has a regression test citing the Lua lines; refrozen fixtures (if any) are listed above with the reason
- [ ] A change of the masks or the action types bumps `ACTION_VERSION`
- [ ] A change users see has a line under "Unreleased" in CHANGELOG.md

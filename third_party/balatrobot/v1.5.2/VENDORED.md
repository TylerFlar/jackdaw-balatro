# Vendored: BalatroBot v1.5.2

| | |
|---|---|
| Upstream | https://github.com/coder/balatrobot |
| Tag | `v1.5.2` (lightweight tag, points straight at the commit) |
| Commit | `9052d76f14723293f6c6b2cecaa791a5c4ae68f3` ("chore(main): release 1.5.2", committed 2026-06-11T17:21:34Z) |
| Tree | `11705bc0d48f21e1cc8d283cda7985ec300b7578` |
| Retrieved | 2026-10-23 |
| License | MIT, Copyright (c) 2025 Coder (see `LICENSE`, vendored unchanged) |

## Purpose

These files are the authority for jackdaw's BalatroBot 1:1 layer (`jackdaw-bot`, used by
`SimBackend`, `Game.bot_view()` and `jackdaw serve`). That layer is ported **endpoint by
endpoint, branch by branch** from the Lua here, and every ported module carries a header pinning
the commit above. In particular:

- `src/lua/core/` (envelope, dispatcher, validator) and `src/lua/endpoints/*.lua` define the
  state gates, parameter validation, and the exact error codes (-32000..-32003) and messages.
- `src/lua/utils/gamestate.lua` builds the view that `bot_view()` must reproduce field for field.
- `src/lua/utils/openrpc.json` is what `rpc.discover` serves.
- `tests/lua/` is the upstream endpoint suite, to be run against `jackdaw serve` (phase 6).
  Every `message =` in the vendored endpoint Lua needs a matching row in jackdaw's error-table
  coverage gate.

## Rules

- **Do not edit anything in this directory.** Patches and deviations belong in jackdaw's own code
  (and, for live behaviour, in the deviation catalog), never here.
- `MANIFEST.sha256` lists every vendored file (sha256sum text format, paths relative to this
  directory, sorted). `tests/bot/test_vendored_balatrobot.py` fails on any hash mismatch,
  missing file, or unlisted file. To check by hand: `cd third_party/balatrobot/v1.5.2 &&
  sha256sum -c MANIFEST.sha256`.
- `../.gitattributes` (`* -text`) stops git from rewriting line endings, so the hashes hold on
  every platform. `../ruff.toml` keeps the repo-wide `ruff check .` / `ruff format --check .`
  away from the upstream Python.
- To upgrade, vendor the new release beside this one (`third_party/balatrobot/vX.Y.Z/`), with its
  own `VENDORED.md` and manifest, and port the differences deliberately.

## How the files were retrieved

The recursive git tree of the pinned commit was listed through the GitHub API, and each selected
blob was downloaded by its blob SHA and checked against the tree's git SHA-1
(`sha1("blob <size>\0" + bytes)`). A sample was also compared with
`raw.githubusercontent.com/coder/balatrobot/<commit>/<path>`. No upstream file contains CRLF line
endings. Binary assets were skipped (the selection contains none).

## File list (74 files, 672,579 bytes)

Requested set:

- `LICENSE`, `README.md`, `docs/api.md`
- `src/lua/settings.lua`
- `src/lua/core/`: `dispatcher.lua`, `server.lua`, `validator.lua`
- `src/lua/endpoints/`: `add.lua`, `buy.lua`, `cash_out.lua`, `discard.lua`, `gamestate.lua`,
  `health.lua`, `load.lua`, `menu.lua`, `next_round.lua`, `pack.lua`, `play.lua`,
  `rearrange.lua`, `reroll.lua`, `save.lua`, `screenshot.lua`, `select.lua`, `sell.lua`,
  `set.lua`, `skip.lua`, `start.lua`, `use.lua`
- `src/lua/endpoints/tests/` (debug-only test endpoints): `echo.lua`, `endpoint.lua`,
  `error.lua`, `state.lua`, `validation.lua`
- `src/lua/utils/`: `debugger.lua`, `enums.lua`, `errors.lua`, `gamestate.lua`, `logger.lua`,
  `openrpc.json`, `types.lua`
- `tests/__init__.py`, `tests/conftest.py` (registers the `integration` marker)
- `tests/fixtures/`: `fixtures.json` (setup steps for every saved-state fixture the suite loads),
  `generate.py` (replays those steps against a live BalatroBot and saves `.jkr` files)
- `tests/lua/`: `__init__.py`, `conftest.py`
- `tests/lua/core/`: `__init__.py`, `test_dispatcher.py`, `test_server.py`, `test_validator.py`
- `tests/lua/endpoints/`: `__init__.py`, `test_add.py`, `test_buy.py`, `test_cash_out.py`,
  `test_discard.py`, `test_gamestate.py`, `test_health.py`, `test_load.py`, `test_menu.py`,
  `test_next_round.py`, `test_pack.py`, `test_play.py`, `test_rearrange.py`, `test_reroll.py`,
  `test_save.py`, `test_screenshot.py`, `test_select.py`, `test_sell.py`, `test_set.py`,
  `test_skip.py`, `test_start.py`, `test_use.py`

Also included, because the Lua layer can't be read correctly without them:

- `balatrobot.lua`: the mod entry point. It defines `BB_ENDPOINTS` (which endpoints are
  registered, in order), loads the test endpoints only in debug mode, and hooks `love.update` to
  run `BB_GAMESTATE.check_game_over()` before each game update.
- `balatrobot.json`: the Steamodded mod manifest. Note that it still says `"version": "1.5.1"` at
  the v1.5.2 tag; upstream did not bump it.

Not vendored: the upstream Python client and CLI (`src/balatrobot/`), `tests/cli/`, the other docs,
SVG assets, and CI/tooling configs.

## Notes for running the upstream suite against `jackdaw serve`

- `tests/lua/conftest.py` imports `balatrobot.config.Config` and
  `balatrobot.manager.BalatroInstance` from the upstream client package, which is not vendored.
  It only uses `Config.from_env()`, `BalatroInstance(config, port=...)` and the async
  `start()`/`stop()`, so a small shim that launches `jackdaw serve --port N` is enough.
- The suite expects `pytest-asyncio` (`asyncio_mode = "auto"`, session loop scope),
  `pytest-xdist` (the `worker_id` fixture; upstream runs with `--dist loadscope`) and `httpx`.
  Upstream's pytest settings live in its `pyproject.toml`, which is not vendored, so the harness
  supplies its own ini.
- `load_fixture` caches generated `.jkr` saves under `tests/fixtures/<endpoint>/`. Run the suite
  from a copy (or redirect the fixture directory), never in place: generated files here would
  fail the manifest test.

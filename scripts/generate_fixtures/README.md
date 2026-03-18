# Fixture Generators

One-time scripts that produce test fixture files in `tests/fixtures/`.

These are **not** runtime tools — they only need to be re-run if the
engine's shop/pool logic changes and fixtures need regenerating.

## Scripts

- `run_shop_oracle.py` — generates `tests/fixtures/shop_oracle_TESTSEED.json`
  by calling `populate_shop` for known seeds. The fixture is loaded by
  `tests/engine/test_shop_oracle.py`.

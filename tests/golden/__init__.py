"""Golden behaviour-equivalence tests (see test_golden_trajectories.py).

Triage for a failing golden test: decide whether the change is behaviour or
storage before touching the fixtures.  Values, RNG draws, sort_id
consumption, dict insertion order (it feeds pseudorandom_element and other
iteration) and aliasing of mutable objects (lists, dicts, ndarrays, cards)
are behaviour.  Attribute storage layout (__dict__ vs __slots__, attribute
order) and sharing of immutable values (tuples, frozen dataclasses) are
not, and the fingerprint already ignores them.  Only regenerate the
fixtures for an intended behaviour change, and say so in the PR.
"""

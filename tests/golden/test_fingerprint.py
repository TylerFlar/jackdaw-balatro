"""Unit tests for the golden fingerprint encoding."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from jackdaw.engine.card_factory import create_joker
from tests.golden.fingerprint import fingerprint


def test_dynamic_dataclass_attributes_count():
    a, b = create_joker("j_joker"), create_joker("j_joker")
    b.sort_id = a.sort_id
    assert fingerprint(a) == fingerprint(b)
    for attr in ("shattered", "lucky_trigger", "vampired"):
        setattr(b, attr, True)
        assert fingerprint(a) != fingerprint(b), attr
        delattr(b, attr)


def test_aliasing_of_mutable_objects_is_encoded():
    x = [1]
    assert fingerprint([x, x]) != fingerprint([x, [1]])
    arr = np.zeros(3)
    assert fingerprint([arr, arr]) != fingerprint([arr, arr.copy()])


def test_immutable_values_encode_by_value():
    t = (1, 2)
    assert fingerprint([t, t]) == fingerprint([t, (1, 2)])

    @dataclasses.dataclass(frozen=True)
    class Frozen:
        v: int

    f = Frozen(1)
    assert fingerprint([f, f]) == fingerprint([f, Frozen(1)])


def test_cycles_terminate():
    d: dict = {}
    d["self"] = d
    lst: list = []
    lst.append(lst)
    assert fingerprint(d) != fingerprint(lst)


def test_attribute_storage_order_is_not_behaviour():
    class P:
        pass

    a, b = P(), P()
    a.x, a.y = 1, 2
    b.y, b.x = 2, 1
    assert fingerprint(a) == fingerprint(b)


def test_dict_order_is_behaviour():
    assert fingerprint({"a": 1, "b": 2}) != fingerprint({"b": 2, "a": 1})


def test_floats_bool_int_are_distinct():
    assert len({fingerprint(1), fingerprint(1.0), fingerprint(True)}) == 3
    assert fingerprint(0.1 + 0.2) != fingerprint(0.3)
    assert fingerprint(float("nan")) == fingerprint(float("nan"))


def test_sets_are_order_independent():
    assert fingerprint({"a", "b", "c"}) == fingerprint({"c", "b", "a"})
    assert fingerprint(frozenset({1, 2})) != fingerprint(frozenset({1, 3}))


def test_unsupported_values_raise():
    with pytest.raises(TypeError):
        fingerprint({object()})
    with pytest.raises(TypeError):
        fingerprint(np.array([object()], dtype=object))
    with pytest.raises(TypeError):
        fingerprint(object())

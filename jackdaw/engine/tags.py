"""Tag class and effect dispatch for all 24 Balatro tags.

Tags are rewards granted when a player skips a blind.  Each tag fires at a
specific game moment (``context``) and returns a :class:`TagResult` describing
the effect, or ``None`` if the context doesn't match.

Source references
-----------------
- ``tag.lua`` — full tag effect dispatch
- ``jackdaw/engine/data/tags.json`` — 24 tag configs

Context types
-------------

+------------------------+-------------------------------------------------------+
| Context                | Tags                                                  |
+========================+=======================================================+
| ``immediate``          | economy, garbage, handy, orbital, skip, top_up        |
+------------------------+-------------------------------------------------------+
| ``new_blind_choice``   | boss, buffoon, charm, ethereal, meteor, standard       |
+------------------------+-------------------------------------------------------+
| ``eval``               | investment (only when last blind was boss)             |
+------------------------+-------------------------------------------------------+
| ``tag_add``            | double (only when added tag != tag_double)             |
+------------------------+-------------------------------------------------------+
| ``round_start_bonus``  | juggle (+3 hand size for one round)                   |
+------------------------+-------------------------------------------------------+
| ``store_joker_create`` | rare (force rarity 3), uncommon (force rarity 2)      |
+------------------------+-------------------------------------------------------+
| ``shop_start``         | d_six (free rerolls this shop)                        |
+------------------------+-------------------------------------------------------+
| ``store_joker_modify`` | foil, holo, polychrome, negative                      |
+------------------------+-------------------------------------------------------+
| ``shop_final_pass``    | coupon (free all shop items)                          |
+------------------------+-------------------------------------------------------+
| ``voucher_add``        | voucher (add a free voucher)                          |
+------------------------+-------------------------------------------------------+

Pack key conventions
--------------------
- Charm Tag:   ``p_arcana_mega_1``  (always variant 1; see rng.py non-determinism note)
- Meteor Tag:  ``p_celestial_mega_1``
- Ethereal Tag: ``p_spectral_normal_1``
- Standard Tag: ``p_standard_mega_1``
- Buffoon Tag:  ``p_buffoon_mega_1``

Blind-tag generation
--------------------

Tags are generated once per ante in ``Game:start_run / reset_blinds``
(``game.lua:2177-2180``).  The call order is strictly:

1. ``get_new_boss()``        — boss blind for this ante
2. ``get_next_voucher_key()`` — voucher for the shop after the boss
3. ``get_next_tag_key()``    — skip reward for the Small blind
4. ``get_next_tag_key()``    — skip reward for the Big blind

Preserving this order is critical for seed determinism.

Public generators
-----------------
:func:`generate_blind_tags` — produce the ``{Small, Big}`` tag dict.
:func:`assign_ante_blinds`  — full ante setup returning blind choices,
    blind tags, and voucher in one call; also updates ``game_state``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from jackdaw.engine.data.hands import HANDS_PAIRS_ORDER, HandType
from jackdaw.engine.data.prototypes import TAGS

if TYPE_CHECKING:
    from jackdaw.engine.rng import PseudoRandom


@dataclass
class TagResult:
    """Descriptor for the effect produced by a :class:`Tag`.

    All fields are optional / zero-valued by default.  Callers inspect
    only the fields relevant to their context.
    """

    dollars: int = 0
    """Immediate cash reward (economy, garbage, handy, skip, investment)."""

    create_pack: str | None = None
    """Pack key to open immediately (buffoon, charm, ethereal, meteor, standard)."""

    create_voucher: bool = False
    """Create a free random voucher from the current pool (voucher tag)."""

    force_rarity: int | None = None
    """Force the next shop joker to this rarity (1=Common, 2=Uncommon, 3=Rare)."""

    force_edition: str | None = None
    """Apply this edition to the shop joker (foil/holo/polychrome/negative)."""

    free_rerolls: int = 0
    """Number of free rerolls (Chaos-style; unused by tags currently)."""

    temp_reroll_zero: bool = False
    """D6 Tag: this shop's rerolls START at $0 and climb (+1 per reroll).
    Vanilla sets round_resets.temp_reroll_cost = 0 and recalcs with
    skip_increment (tag.lua:383-391), once per shop (shop_d6ed guard)."""

    double: bool = False
    """Double Tag: duplicate the tag that was just added."""

    level_up: tuple[HandType, int] | None = None
    """(hand_type, levels) — level up a hand type N times (orbital)."""

    reroll_boss: bool = False
    """Reroll the current boss blind (boss tag)."""

    hand_size_delta: int = 0
    """Temporary hand-size increase for this round (juggle)."""

    create_jokers: int = 0
    """Number of Common jokers to add for free (top-up, up to joker slot cap)."""

    coupon: bool = False
    """Make all shop items free (coupon tag)."""


# The ``pairs(G.GAME.hands)`` iteration order (HANDS_PAIRS_ORDER, G182): the
# Orbital Tag (UI_definitions.lua:1510-1515) and To Do List
# (card.lua:313-320) both collect the visible hands with ``pairs`` into an
# array, and pseudorandom_element indexes that array as built.
_ORBITAL_HANDS: list[HandType] = list(HANDS_PAIRS_ORDER)


class Tag:
    """Runtime representation of a Balatro tag.

    Parameters
    ----------
    key:
        Tag key from ``TAGS`` (e.g. ``"tag_economy"``).
    tag_id:
        Optional integer identifier used by the game to track which specific
        tag instance triggered.  Not required for effect dispatch.
    """

    __slots__ = ("key", "proto", "name", "config", "id", "triggered", "orbital_hand")

    def __init__(
        self, key: str, tag_id: int | None = None, *, orbital_hand: str | None = None
    ) -> None:
        self.key: str = key
        self.proto = TAGS[key]
        self.name: str = self.proto.name
        self.config: dict[str, Any] = dict(self.proto.config) if self.proto.config else {}
        self.id: int | None = tag_id
        self.triggered: bool = False
        # ability.orbital_hand (tag.lua:103-112): the blind's pre-rolled
        # Orbital hand, or the original's for a Double Tag copy.
        self.orbital_hand: str | None = orbital_hand

    # ------------------------------------------------------------------
    # Effect dispatch
    # ------------------------------------------------------------------

    def apply(
        self,
        context: str,
        game_state: dict[str, Any],
        rng: PseudoRandom | None = None,
        **kwargs: Any,
    ) -> TagResult | None:
        """Fire the tag's effect.

        Returns a :class:`TagResult` when the tag fires, ``None`` when the
        context does not match (tag does not apply here).

        Parameters
        ----------
        context:
            The game moment string (e.g. ``"immediate"``, ``"eval"``, …).
        game_state:
            Mutable run-state dict.  Some tags read values from this dict
            (e.g. ``game_state.get("dollars")`` for Economy Tag).
        rng:
            Active :class:`~jackdaw.engine.rng.PseudoRandom` instance.
            Required for Orbital Tag's random hand selection.
        **kwargs:
            Context-specific extra data:

            * ``last_blind_is_boss`` (bool) — Investment Tag
            * ``added_tag_key`` (str)       — Double Tag
        """
        tag_type = self.config.get("type", "")

        # ----------------------------------------------------------------
        # immediate — fires right when the tag is collected
        # ----------------------------------------------------------------
        if context == "immediate":
            if tag_type != "immediate":
                return None

            if self.key == "tag_economy":
                current = game_state.get("dollars", 0)
                reward = min(self.config["max"], max(0, current))
                return TagResult(dollars=reward)

            if self.key == "tag_garbage":
                unused = game_state.get("unused_discards", 0)
                return TagResult(dollars=unused * self.config["dollars_per_discard"])

            if self.key == "tag_handy":
                played = game_state.get("hands_played", 0)
                return TagResult(dollars=played * self.config["dollars_per_hand"])

            if self.key == "tag_skip":
                skips = game_state.get("skips", 0)
                return TagResult(dollars=skips * self.config["skip_bonus"])

            if self.key == "tag_top_up":
                return TagResult(create_jokers=self.config["spawn_jokers"])

            if self.key == "tag_orbital":
                # Levels the hand its blind rolled when the blind-select
                # screen was built (tag.lua:191-197; see
                # ensure_orbital_choices).  A bare Tag outside that flow
                # rolls one the same way.
                hand = self.orbital_hand
                if hand is None:
                    if rng is None:
                        raise ValueError("tag_orbital requires an rng instance")
                    hand = _roll_orbital_hand(rng, game_state.get("hand_levels"))
                levels = self.config["levels"]
                return TagResult(level_up=(HandType(hand), levels))

            # Unknown immediate tag — no effect
            return None  # pragma: no cover

        # ----------------------------------------------------------------
        # new_blind_choice — fires when entering a new blind (skip reward)
        # ----------------------------------------------------------------
        if context == "new_blind_choice":
            if tag_type != "new_blind_choice":
                return None

            if self.key == "tag_boss":
                return TagResult(reroll_boss=True)

            if self.key == "tag_buffoon":
                return TagResult(create_pack="p_buffoon_mega_1")

            if self.key == "tag_charm":
                # Lua uses math.random(1,2) here — non-deterministic.
                # We always pick variant 1 (see rng.py Known Non-Determinism).
                return TagResult(create_pack="p_arcana_mega_1")

            if self.key == "tag_ethereal":
                return TagResult(create_pack="p_spectral_normal_1")

            if self.key == "tag_meteor":
                # Same non-determinism caveat as Charm — always variant 1.
                return TagResult(create_pack="p_celestial_mega_1")

            if self.key == "tag_standard":
                return TagResult(create_pack="p_standard_mega_1")

            return None  # pragma: no cover

        # ----------------------------------------------------------------
        # eval — fires after each blind is scored
        # ----------------------------------------------------------------
        if context == "eval":
            if tag_type != "eval":
                return None

            if self.key == "tag_investment":
                if kwargs.get("last_blind_is_boss", False):
                    return TagResult(dollars=self.config["dollars"])
            return None

        # ----------------------------------------------------------------
        # tag_add — fires when any tag is added to the run
        # ----------------------------------------------------------------
        if context == "tag_add":
            if tag_type != "tag_add":
                return None

            if self.key == "tag_double":
                added_key = kwargs.get("added_tag_key", "")
                if added_key and added_key != "tag_double":
                    return TagResult(double=True)
            return None

        # ----------------------------------------------------------------
        # round_start_bonus — fires at the start of each round
        # ----------------------------------------------------------------
        if context == "round_start_bonus":
            if tag_type != "round_start_bonus":
                return None

            if self.key == "tag_juggle":
                return TagResult(hand_size_delta=self.config["h_size"])

            return None  # pragma: no cover

        # ----------------------------------------------------------------
        # store_joker_create — fires when a shop joker slot is being filled
        # ----------------------------------------------------------------
        if context == "store_joker_create":
            if tag_type != "store_joker_create":
                return None

            if self.key == "tag_uncommon":
                return TagResult(force_rarity=2)

            if self.key == "tag_rare":
                return TagResult(force_rarity=3)

            return None  # pragma: no cover

        # ----------------------------------------------------------------
        # shop_start — fires at the beginning of the shop phase
        # ----------------------------------------------------------------
        if context == "shop_start":
            if tag_type != "shop_start":
                return None

            if self.key == "tag_d_six":
                return TagResult(temp_reroll_zero=True)

            return None  # pragma: no cover

        # ----------------------------------------------------------------
        # store_joker_modify — fires when a shop joker exists, adds edition
        # ----------------------------------------------------------------
        if context == "store_joker_modify":
            if tag_type != "store_joker_modify":
                return None

            edition = self.config.get("edition")
            if edition:
                return TagResult(force_edition=edition)

            return None  # pragma: no cover

        # ----------------------------------------------------------------
        # shop_final_pass — fires at the end of shop generation
        # ----------------------------------------------------------------
        if context == "shop_final_pass":
            if tag_type != "shop_final_pass":
                return None

            if self.key == "tag_coupon":
                return TagResult(coupon=True)

            return None  # pragma: no cover

        # ----------------------------------------------------------------
        # voucher_add — fires when a voucher is being added to the shop
        # ----------------------------------------------------------------
        if context == "voucher_add":
            if tag_type != "voucher_add":
                return None

            if self.key == "tag_voucher":
                return TagResult(create_voucher=True)

            return None  # pragma: no cover

        # Unknown context — tag does not apply
        return None

    def __repr__(self) -> str:
        return f"Tag({self.key!r}, id={self.id})"


# ---------------------------------------------------------------------------
# Orbital Tag hands
# ---------------------------------------------------------------------------

_BLIND_TYPES: tuple[str, ...] = ("Small", "Big", "Boss")


def _roll_orbital_hand(rng: PseudoRandom, hand_levels: Any) -> str:
    """One ``'orbital'`` pick over the visible hands (UI_definitions.lua:1510-1515)."""
    pool = [
        ht.value
        for ht in _ORBITAL_HANDS
        if hand_levels is None or hand_levels.get_state(ht).visible
    ]
    hand, _ = rng.element(pool, rng.seed("orbital"))
    return hand


def ensure_orbital_choices(game_state: dict[str, Any]) -> dict[str, str]:
    """The current ante's Orbital Tag hands, rolled the first time they are needed.

    create_UIBox_blind_choice (UI_definitions.lua:1506-1516) is called for
    Small, Big and Boss, in that order, whenever the blind-select screen is
    built (run start, every return from the shop: game.lua:3258-3298).  For
    each blind type it fills ``G.GAME.orbital_choices[ante][type]`` if
    unset, with a ``pseudorandom_element`` over the hands visible at that
    moment (stream ``'orbital'``).  So each ante rolls three hands at its
    first blind-select screen; later screens in the ante (after a shop, a
    boss reroll) reuse them, and an ante revisited after Hieroglyph keeps
    its old ones.  The skip-tag Tag objects are made right after, so an
    Orbital Tag carries its blind's hand (tag.lua:107-110).
    """
    ante = game_state["round_resets"]["ante"]
    choices = game_state.setdefault("orbital_choices", {}).setdefault(ante, {})
    for blind_type in _BLIND_TYPES:
        if blind_type not in choices:
            choices[blind_type] = _roll_orbital_hand(
                game_state["rng"], game_state.get("hand_levels")
            )
    return choices


# ---------------------------------------------------------------------------
# Blind-tag generation
# ---------------------------------------------------------------------------


def generate_blind_tags(
    ante: int,
    rng: PseudoRandom,
    game_state: dict[str, Any],
) -> dict[str, str]:
    """Generate tags for Small and Big blind skip rewards.

    Calls :func:`~jackdaw.engine.pools.pick_card_from_pool` twice with the
    ``"Tag"`` pool, applying ``min_ante`` and ``requires`` filters from
    *game_state*.

    Parameters
    ----------
    ante:
        Current ante number (1–8).
    rng:
        Live :class:`~jackdaw.engine.rng.PseudoRandom` instance.  Each
        call advances the ``"Tag{ante}"`` stream once.
    game_state:
        Run-state dict.  Reads:

        * ``used_vouchers`` (iterable of str, optional) — purchased voucher
          keys; used to gate tags with a ``requires`` field.

    Returns
    -------
    dict
        ``{"Small": tag_key, "Big": tag_key}``
    """
    from jackdaw.engine.pools import pick_card_from_pool

    used_vouchers: set[str] = set(game_state.get("used_vouchers", []))
    discovered: set[str] | None = game_state.get("discovered")
    banned = game_state.get("banned_keys")
    small = pick_card_from_pool(
        "Tag", rng, ante, used_vouchers=used_vouchers, discovered=discovered, banned_keys=banned
    )
    big = pick_card_from_pool(
        "Tag", rng, ante, used_vouchers=used_vouchers, discovered=discovered, banned_keys=banned
    )
    return {"Small": small, "Big": big}


def roll_next_voucher(ante: int, rng: PseudoRandom, game_state: dict[str, Any]) -> str | None:
    """``get_next_voucher_key()`` for the ante's shop voucher.

    Lua rolls it at run start (game.lua:2177) and when a boss falls, in
    end_round's last event (state_events.lua:263).  By then the shop is gone
    (``G.shop:remove()``, button_callbacks.lua:2500, leaves
    ``G.shop_vouchers.cards`` nil, cardarea.lua:659), so get_current_pool's
    shop exclusion (common_events.lua:1999-2003) never applies.
    """
    from jackdaw.engine.vouchers import get_next_voucher_key

    used_v_dict = {k: True for k in set(game_state.get("used_vouchers", []))}
    banned: dict[str, bool] = game_state.get("banned_keys") or {}
    return get_next_voucher_key(rng, used_v_dict, in_shop=None, ante=ante, banned_keys=banned)


def roll_ante_blinds(ante: int, rng: PseudoRandom, game_state: dict[str, Any]) -> str:
    """The ante's blind tags and boss: ``get_next_tag_key()`` for Small, then
    Big (button_callbacks.lua:2949-2952), then ``reset_blinds``'s
    ``get_new_boss()`` (common_events.lua:2333).  Lua runs them at run start
    and in ``cash_out`` after a boss.  Sets ``round_resets.blind_tags`` and
    returns the boss key.
    """
    from jackdaw.engine.blind import get_new_boss
    from jackdaw.engine.pools import pick_card_from_pool

    bosses_used: dict[str, int] = game_state.setdefault("bosses_used", {})
    used_vouchers: set[str] = set(game_state.get("used_vouchers", []))
    # Challenge bans cull the tag and boss pools (common_events.lua:2030, 2359-2361).
    banned: dict[str, bool] = game_state.get("banned_keys") or {}
    small = pick_card_from_pool("Tag", rng, ante, used_vouchers=used_vouchers, banned_keys=banned)
    big = pick_card_from_pool("Tag", rng, ante, used_vouchers=used_vouchers, banned_keys=banned)
    boss = get_new_boss(
        ante,
        bosses_used,
        rng,
        win_ante=game_state.get("win_ante", 8),
        banned_keys=banned,
    )
    game_state.setdefault("round_resets", {})["blind_tags"] = {"Small": small, "Big": big}
    return boss


def assign_ante_blinds(
    ante: int,
    rng: PseudoRandom,
    game_state: dict[str, Any],
) -> dict[str, Any]:
    """The first ante's voucher, tags and boss, in Lua's run-start order
    (:func:`roll_next_voucher`, then :func:`roll_ante_blinds`).

    Returns ``{"blind_choices": {"Small": "bl_small", "Big": "bl_big",
    "Boss": boss_key}, "blind_tags": {"Small": tag_key, "Big": tag_key},
    "voucher": voucher_key_or_None}``.
    """
    voucher = roll_next_voucher(ante, rng, game_state)
    boss = roll_ante_blinds(ante, rng, game_state)
    return {
        "blind_choices": {"Small": "bl_small", "Big": "bl_big", "Boss": boss},
        "blind_tags": dict(game_state["round_resets"]["blind_tags"]),
        "voucher": voucher,
    }

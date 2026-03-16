"""Joker evaluation dispatch system.

Ports ``Card:calculate_joker`` from ``card.lua:2291`` — a 1,770-line
if/elseif chain — as a dispatch table keyed by center key.

Each joker handler is a pure function ``(Card, JokerContext) -> JokerResult | None``
registered via the ``@register`` decorator.  The ``calculate_joker`` entry point
checks the debuff flag, looks up the handler, and dispatches.

Source: card.lua:2291-4060 (calculate_joker), common_events.lua:571-1065 (call sites).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from jackdaw.engine.blind import Blind
    from jackdaw.engine.card import Card
    from jackdaw.engine.hand_levels import HandLevels
    from jackdaw.engine.rng import PseudoRandom


# ---------------------------------------------------------------------------
# JokerContext — mirrors the ``context`` table passed to calculate_joker
# ---------------------------------------------------------------------------

@dataclass
class JokerContext:
    """Context table passed to each joker handler.

    Exactly one phase flag should be set per call, matching the source's
    context-based branching in card.lua:2291+.
    """

    # Phase flags (exactly one set per call)
    before: bool = False
    individual: bool = False
    repetition: bool = False
    joker_main: bool = False
    after: bool = False
    other_joker: Card | None = None
    setting_blind: bool = False
    first_hand_drawn: bool = False
    end_of_round: bool = False
    discard: bool = False
    pre_discard: bool = False
    destroying_card: Card | None = None
    cards_destroyed: list[Card] | None = None
    selling_self: bool = False
    selling_card: bool = False
    open_booster: bool = False
    skip_blind: bool = False
    reroll_shop: bool = False
    ending_shop: bool = False
    debuffed_hand: bool = False
    using_consumeable: bool = False
    playing_card_added: bool = False

    # Context data
    cardarea: str | None = None
    other_card: Card | None = None
    full_hand: list[Card] | None = None
    scoring_hand: list[Card] | None = None
    scoring_name: str | None = None
    poker_hands: dict[str, Any] | None = None
    blueprint: int = 0
    rng: PseudoRandom | None = None
    hand_levels: HandLevels | None = None
    smeared: bool = False
    pareidolia: bool = False
    probabilities_normal: float = 1.0
    ancient_suit: str | None = None
    idol_card: dict[str, Any] | None = None

    # Game state (pre-computed by scoring pipeline)
    joker_count: int = 0
    joker_slots: int = 5
    hands_left: int = 0
    discards_left: int = 0
    deck_cards_remaining: int = 0
    starting_deck_size: int = 52
    playing_cards_count: int = 52
    stone_tally: int = 0
    steel_tally: int = 0
    money: int = 0
    enhanced_card_count: int = 0
    consumable_usage_tarot: int = 0
    hands_played: int = 0
    held_cards: list[Card] | None = None
    mail_card_id: int | None = None
    discards_used: int = 0
    blind: Blind | None = None
    jokers: list[Card] | None = None
    blueprint_card: Card | None = None


# ---------------------------------------------------------------------------
# JokerResult — return value from joker handlers
# ---------------------------------------------------------------------------

@dataclass
class JokerResult:
    """Return value from a joker handler.

    Fields match BOTH return formats used by the source:
    - ``individual`` context returns: chips, mult, x_mult, dollars
    - ``joker_main`` context returns: chip_mod, mult_mod, Xmult_mod
    - ``repetition`` context returns: repetitions
    """

    # Individual context returns
    chips: float = 0
    mult: float = 0
    x_mult: float = 0
    h_mult: float = 0  # held-card mult (Raised Fist, Shoot the Moon)
    dollars: int = 0

    # Joker main context returns (different naming!)
    chip_mod: float = 0
    mult_mod: float = 0
    Xmult_mod: float = 0  # noqa: N815 — matches Lua naming

    # Repetition context
    repetitions: int = 0

    # State change signals
    level_up: bool = False
    saved: bool = False
    remove: bool = False
    message: str = ""

    # Extra (for unusual effects like Perkeo, Showman, etc.)
    extra: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Handler registry
# ---------------------------------------------------------------------------

JokerHandler = Callable[["Card", JokerContext], JokerResult | None]
_REGISTRY: dict[str, JokerHandler] = {}


def register(key: str) -> Callable[[JokerHandler], JokerHandler]:
    """Decorator to register a joker handler by center key.

    Usage::

        @register("j_joker")
        def _joker(card: Card, ctx: JokerContext) -> JokerResult | None:
            if ctx.joker_main:
                return JokerResult(mult_mod=card.ability["extra"])
            return None
    """
    def wrapper(fn: JokerHandler) -> JokerHandler:
        _REGISTRY[key] = fn
        return fn
    return wrapper


def calculate_joker(card: Card, context: JokerContext) -> JokerResult | None:
    """Main dispatch — mirrors Card:calculate_joker from card.lua:2291.

    Returns ``None`` if the joker is debuffed, unregistered, or has no
    effect in the given context.
    """
    if card.debuff:
        return None
    handler = _REGISTRY.get(card.center_key)
    if handler is None:
        return None
    return handler(card, context)


def registered_jokers() -> list[str]:
    """Return sorted list of all registered joker center keys."""
    return sorted(_REGISTRY)


# ---------------------------------------------------------------------------
# Joker handlers — simple unconditional bonuses
# ---------------------------------------------------------------------------

@register("j_joker")
def _joker(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Joker: +mult (default 4). Source: card.lua:3980."""
    if ctx.joker_main:
        return JokerResult(mult_mod=card.ability.get("mult", 4))
    return None


@register("j_misprint")
def _misprint(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Misprint: random mult between extra.min and extra.max each hand.

    Source: card.lua:3700 — ``pseudorandom('misprint', min, max)``.
    Rolls a fresh value every hand via the 'misprint' RNG stream.
    """
    if ctx.joker_main:
        extra = card.ability.get("extra", {})
        lo = extra.get("min", 0)
        hi = extra.get("max", 23)
        if ctx.rng is not None:
            roll = ctx.rng.random("misprint", min_val=lo, max_val=hi)
        else:
            roll = 0
        return JokerResult(mult_mod=roll)
    return None


@register("j_stuntman")
def _stuntman(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Stuntman: +chip_mod chips (default 250). Source: card.lua:3713."""
    if ctx.joker_main:
        extra = card.ability.get("extra", {})
        return JokerResult(chip_mod=extra.get("chip_mod", 250))
    return None


# ---------------------------------------------------------------------------
# Helper: poker_hands containment check
# ---------------------------------------------------------------------------

def _contains(ctx: JokerContext, hand_type: str) -> bool:
    """Check if poker_hands contains a non-empty entry for *hand_type*.

    Equivalent to Lua's ``next(context.poker_hands[type])``.
    """
    if ctx.poker_hands is None:
        return False
    entries = ctx.poker_hands.get(hand_type)
    return bool(entries)


# ---------------------------------------------------------------------------
# Joker handlers — hand-type mult bonuses (Category A pattern, but uses
# poker_hands containment check per source card.lua:3660)
# ---------------------------------------------------------------------------

@register("j_jolly")
def _jolly(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Jolly Joker: +t_mult if hand contains Pair. Source: card.lua:3660."""
    if ctx.joker_main and _contains(ctx, card.ability.get("type", "Pair")):
        return JokerResult(mult_mod=card.ability.get("t_mult", 8))
    return None


@register("j_zany")
def _zany(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Zany Joker: +t_mult if hand contains Three of a Kind. Source: card.lua:3660."""
    if ctx.joker_main and _contains(ctx, card.ability.get("type", "Three of a Kind")):
        return JokerResult(mult_mod=card.ability.get("t_mult", 12))
    return None


@register("j_mad")
def _mad(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Mad Joker: +t_mult if hand contains Two Pair. Source: card.lua:3660."""
    if ctx.joker_main and _contains(ctx, card.ability.get("type", "Two Pair")):
        return JokerResult(mult_mod=card.ability.get("t_mult", 10))
    return None


@register("j_crazy")
def _crazy(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Crazy Joker: +t_mult if hand contains Straight. Source: card.lua:3660."""
    if ctx.joker_main and _contains(ctx, card.ability.get("type", "Straight")):
        return JokerResult(mult_mod=card.ability.get("t_mult", 12))
    return None


@register("j_droll")
def _droll(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Droll Joker: +t_mult if hand contains Flush. Source: card.lua:3660."""
    if ctx.joker_main and _contains(ctx, card.ability.get("type", "Flush")):
        return JokerResult(mult_mod=card.ability.get("t_mult", 10))
    return None


# ---------------------------------------------------------------------------
# Joker handlers — hand-type chip bonuses (poker_hands containment check)
# Source: card.lua:3666
# ---------------------------------------------------------------------------

@register("j_sly")
def _sly(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Sly Joker: +t_chips if hand contains Pair."""
    if ctx.joker_main and _contains(ctx, card.ability.get("type", "Pair")):
        return JokerResult(chip_mod=card.ability.get("t_chips", 50))
    return None


@register("j_wily")
def _wily(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Wily Joker: +t_chips if hand contains Three of a Kind."""
    if ctx.joker_main and _contains(ctx, card.ability.get("type", "Three of a Kind")):
        return JokerResult(chip_mod=card.ability.get("t_chips", 100))
    return None


@register("j_clever")
def _clever(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Clever Joker: +t_chips if hand contains Two Pair."""
    if ctx.joker_main and _contains(ctx, card.ability.get("type", "Two Pair")):
        return JokerResult(chip_mod=card.ability.get("t_chips", 80))
    return None


@register("j_devious")
def _devious(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Devious Joker: +t_chips if hand contains Straight."""
    if ctx.joker_main and _contains(ctx, card.ability.get("type", "Straight")):
        return JokerResult(chip_mod=card.ability.get("t_chips", 100))
    return None


@register("j_crafty")
def _crafty(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Crafty Joker: +t_chips if hand contains Flush."""
    if ctx.joker_main and _contains(ctx, card.ability.get("type", "Flush")):
        return JokerResult(chip_mod=card.ability.get("t_chips", 80))
    return None


# ---------------------------------------------------------------------------
# Joker handlers — hand-type xMult bonuses (poker_hands containment check)
# Source: card.lua:3653
# ---------------------------------------------------------------------------

@register("j_duo")
def _duo(card: Card, ctx: JokerContext) -> JokerResult | None:
    """The Duo: xMult if hand contains Pair."""
    if ctx.joker_main and _contains(ctx, card.ability.get("type", "Pair")):
        x = card.ability.get("x_mult", 2)
        if x > 1:
            return JokerResult(Xmult_mod=x)
    return None


@register("j_trio")
def _trio(card: Card, ctx: JokerContext) -> JokerResult | None:
    """The Trio: xMult if hand contains Three of a Kind."""
    if ctx.joker_main and _contains(ctx, card.ability.get("type", "Three of a Kind")):
        x = card.ability.get("x_mult", 3)
        if x > 1:
            return JokerResult(Xmult_mod=x)
    return None


@register("j_family")
def _family(card: Card, ctx: JokerContext) -> JokerResult | None:
    """The Family: xMult if hand contains Four of a Kind."""
    if ctx.joker_main and _contains(ctx, card.ability.get("type", "Four of a Kind")):
        x = card.ability.get("x_mult", 4)
        if x > 1:
            return JokerResult(Xmult_mod=x)
    return None


@register("j_order")
def _order(card: Card, ctx: JokerContext) -> JokerResult | None:
    """The Order: xMult if hand contains Straight."""
    if ctx.joker_main and _contains(ctx, card.ability.get("type", "Straight")):
        x = card.ability.get("x_mult", 3)
        if x > 1:
            return JokerResult(Xmult_mod=x)
    return None


@register("j_tribe")
def _tribe(card: Card, ctx: JokerContext) -> JokerResult | None:
    """The Tribe: xMult if hand contains Flush."""
    if ctx.joker_main and _contains(ctx, card.ability.get("type", "Flush")):
        x = card.ability.get("x_mult", 2)
        if x > 1:
            return JokerResult(Xmult_mod=x)
    return None


# ---------------------------------------------------------------------------
# Joker handlers — scoring_name-based (exact hand type match)
# ---------------------------------------------------------------------------

@register("j_supernova")
def _supernova(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Supernova: +mult = times this hand type played in the run.

    Source: card.lua:3731 — uses ``G.GAME.hands[context.scoring_name].played``.
    """
    if ctx.joker_main and ctx.scoring_name and ctx.hand_levels is not None:
        played = ctx.hand_levels[ctx.scoring_name].played
        return JokerResult(mult_mod=played)
    return None


@register("j_card_sharp")
def _card_sharp(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Card Sharp: xMult if same hand type played twice this round.

    Source: card.lua:4040 — checks ``G.GAME.hands[scoring_name].played_this_round > 1``.
    """
    if ctx.joker_main and ctx.scoring_name and ctx.hand_levels is not None:
        ptr = ctx.hand_levels[ctx.scoring_name].played_this_round
        if ptr > 1:
            extra = card.ability.get("extra", {})
            return JokerResult(Xmult_mod=extra.get("Xmult", 3))
    return None


# ---------------------------------------------------------------------------
# Helper: suit check on other_card
# ---------------------------------------------------------------------------

def _is_suit(ctx: JokerContext, suit: str) -> bool:
    """Check if ctx.other_card matches *suit* using Card.is_suit.

    Passes ``smeared`` from context. Wild Cards match any suit automatically
    via Card.is_suit internals.
    """
    if ctx.other_card is None:
        return False
    return ctx.other_card.is_suit(suit, smeared=ctx.smeared)


# ---------------------------------------------------------------------------
# Joker handlers — suit-conditional per scored card (individual context)
# Source: card.lua:3065 (Suit Mult), 3224-3260 (named suit jokers)
# ---------------------------------------------------------------------------

@register("j_greedy_joker")
def _greedy_joker(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Greedy Joker: +s_mult per Diamond scored. Source: card.lua:3065."""
    if ctx.individual and ctx.cardarea == "play":
        extra = card.ability.get("extra", {})
        if _is_suit(ctx, extra.get("suit", "Diamonds")):
            return JokerResult(mult=extra.get("s_mult", 3))
    return None


@register("j_lusty_joker")
def _lusty_joker(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Lusty Joker: +s_mult per Heart scored. Source: card.lua:3065."""
    if ctx.individual and ctx.cardarea == "play":
        extra = card.ability.get("extra", {})
        if _is_suit(ctx, extra.get("suit", "Hearts")):
            return JokerResult(mult=extra.get("s_mult", 3))
    return None


@register("j_wrathful_joker")
def _wrathful_joker(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Wrathful Joker: +s_mult per Spade scored. Source: card.lua:3065."""
    if ctx.individual and ctx.cardarea == "play":
        extra = card.ability.get("extra", {})
        if _is_suit(ctx, extra.get("suit", "Spades")):
            return JokerResult(mult=extra.get("s_mult", 3))
    return None


@register("j_gluttenous_joker")
def _gluttenous_joker(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Gluttonous Joker: +s_mult per Club scored. Source: card.lua:3065.

    Note: center key has typo ``j_gluttenous_joker`` (double t) matching source.
    """
    if ctx.individual and ctx.cardarea == "play":
        extra = card.ability.get("extra", {})
        if _is_suit(ctx, extra.get("suit", "Clubs")):
            return JokerResult(mult=extra.get("s_mult", 3))
    return None


@register("j_arrowhead")
def _arrowhead(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Arrowhead: +chips per Spade scored. Source: card.lua:3240."""
    if ctx.individual and ctx.cardarea == "play":
        if _is_suit(ctx, "Spades"):
            return JokerResult(chips=card.ability.get("extra", 50))
    return None


@register("j_onyx_agate")
def _onyx_agate(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Onyx Agate: +mult per Club scored. Source: card.lua:3233."""
    if ctx.individual and ctx.cardarea == "play":
        if _is_suit(ctx, "Clubs"):
            return JokerResult(mult=card.ability.get("extra", 7))
    return None


@register("j_rough_gem")
def _rough_gem(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Rough Gem: +$1 per Diamond scored. Source: card.lua:3224."""
    if ctx.individual and ctx.cardarea == "play":
        if _is_suit(ctx, "Diamonds"):
            return JokerResult(dollars=card.ability.get("extra", 1))
    return None


@register("j_bloodstone")
def _bloodstone(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Bloodstone: probabilistic xMult per Heart scored.

    Source: card.lua:3247 — ``pseudorandom('bloodstone') < normal/odds``.
    """
    if ctx.individual and ctx.cardarea == "play":
        if _is_suit(ctx, "Hearts"):
            extra = card.ability.get("extra", {})
            odds = extra.get("odds", 2)
            if ctx.rng is not None:
                roll = ctx.rng.random("bloodstone")
                if roll < ctx.probabilities_normal / odds:
                    return JokerResult(x_mult=extra.get("Xmult", 1.5))
            # No RNG → no effect (same as Lucky Card)
    return None


@register("j_ancient")
def _ancient(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Ancient Joker: xMult if scored card matches current round's ancient suit.

    Source: card.lua:3255 — uses ``G.GAME.current_round.ancient_card.suit``.
    """
    if ctx.individual and ctx.cardarea == "play":
        if ctx.ancient_suit and _is_suit(ctx, ctx.ancient_suit):
            return JokerResult(x_mult=card.ability.get("extra", 1.5))
    return None


# ---------------------------------------------------------------------------
# Joker handlers — rank-conditional per scored card (individual context)
# Source: card.lua:3065-3270
# ---------------------------------------------------------------------------

_FIBONACCI_IDS = frozenset({2, 3, 5, 8, 14})  # Ace=14


@register("j_fibonacci")
def _fibonacci(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Fibonacci: +mult for Ace/2/3/5/8. Source: card.lua:3185."""
    if ctx.individual and ctx.cardarea == "play" and ctx.other_card is not None:
        if ctx.other_card.get_id() in _FIBONACCI_IDS:
            return JokerResult(mult=card.ability.get("extra", 8))
    return None


@register("j_scholar")
def _scholar(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Scholar: +chips and +mult for Ace. Source: card.lua:3159."""
    if ctx.individual and ctx.cardarea == "play" and ctx.other_card is not None:
        if ctx.other_card.get_id() == 14:
            extra = card.ability.get("extra", {})
            return JokerResult(
                chips=extra.get("chips", 20),
                mult=extra.get("mult", 4),
            )
    return None


@register("j_walkie_talkie")
def _walkie_talkie(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Walkie Talkie: +chips and +mult for 10 or 4. Source: card.lua:3167."""
    if ctx.individual and ctx.cardarea == "play" and ctx.other_card is not None:
        oid = ctx.other_card.get_id()
        if oid == 10 or oid == 4:
            extra = card.ability.get("extra", {})
            return JokerResult(
                chips=extra.get("chips", 10),
                mult=extra.get("mult", 4),
            )
    return None


@register("j_even_steven")
def _even_steven(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Even Steven: +mult for even numbered cards (2/4/6/8/10). Source: card.lua:3196."""
    if ctx.individual and ctx.cardarea == "play" and ctx.other_card is not None:
        oid = ctx.other_card.get_id()
        if 0 <= oid <= 10 and oid % 2 == 0:
            return JokerResult(mult=card.ability.get("extra", 4))
    return None


@register("j_odd_todd")
def _odd_todd(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Odd Todd: +chips for odd numbered cards (3/5/7/9) and Ace. Source: card.lua:3206."""
    if ctx.individual and ctx.cardarea == "play" and ctx.other_card is not None:
        oid = ctx.other_card.get_id()
        if (0 <= oid <= 10 and oid % 2 == 1) or oid == 14:
            return JokerResult(chips=card.ability.get("extra", 31))
    return None


@register("j_scary_face")
def _scary_face(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Scary Face: +chips for face cards. Source: card.lua:3136."""
    if ctx.individual and ctx.cardarea == "play" and ctx.other_card is not None:
        if ctx.other_card.is_face(pareidolia=ctx.pareidolia):
            return JokerResult(chips=card.ability.get("extra", 30))
    return None


@register("j_smiley")
def _smiley(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Smiley Face: +mult for face cards. Source: card.lua:3143."""
    if ctx.individual and ctx.cardarea == "play" and ctx.other_card is not None:
        if ctx.other_card.is_face(pareidolia=ctx.pareidolia):
            return JokerResult(mult=card.ability.get("extra", 5))
    return None


@register("j_photograph")
def _photograph(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Photograph: xMult on FIRST face card in scoring hand only.

    Source: card.lua:3093 — loops scoring_hand to find first is_face(),
    then checks ``context.other_card == first_face`` by identity.
    """
    if ctx.individual and ctx.cardarea == "play" and ctx.other_card is not None:
        if ctx.scoring_hand:
            first_face = None
            for sc in ctx.scoring_hand:
                if sc.is_face(pareidolia=ctx.pareidolia):
                    first_face = sc
                    break
            if ctx.other_card is first_face:
                return JokerResult(x_mult=card.ability.get("extra", 2))
    return None


@register("j_triboulet")
def _triboulet(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Triboulet: x2 mult for King or Queen. Source: card.lua:3262."""
    if ctx.individual and ctx.cardarea == "play" and ctx.other_card is not None:
        oid = ctx.other_card.get_id()
        if oid == 12 or oid == 13:
            return JokerResult(x_mult=card.ability.get("extra", 2))
    return None


@register("j_idol")
def _idol(card: Card, ctx: JokerContext) -> JokerResult | None:
    """The Idol: xMult if scored card matches idol_card rank AND suit.

    Source: card.lua:3127 — checks both get_id() and is_suit().
    """
    if ctx.individual and ctx.cardarea == "play" and ctx.other_card is not None:
        if ctx.idol_card is not None:
            if (
                ctx.other_card.get_id() == ctx.idol_card.get("id")
                and _is_suit(ctx, ctx.idol_card.get("suit", ""))
            ):
                return JokerResult(x_mult=card.ability.get("extra", 2))
    return None


# ---------------------------------------------------------------------------
# Joker handlers — rank-conditional retrigger (repetition context)
# Source: card.lua:3374
# ---------------------------------------------------------------------------

_HACK_IDS = frozenset({2, 3, 4, 5})


@register("j_hack")
def _hack(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Hack: retrigger 2/3/4/5 cards. Source: card.lua:3374."""
    if ctx.repetition and ctx.cardarea == "play" and ctx.other_card is not None:
        if ctx.other_card.get_id() in _HACK_IDS:
            return JokerResult(repetitions=card.ability.get("extra", 1))
    return None


# ---------------------------------------------------------------------------
# Joker handlers — game-state-dependent (joker_main context)
# ---------------------------------------------------------------------------

@register("j_half")
def _half(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Half Joker: +mult if ≤3 cards played. Source: card.lua:3672."""
    if ctx.joker_main and ctx.full_hand is not None:
        extra = card.ability.get("extra", {})
        if len(ctx.full_hand) <= extra.get("size", 3):
            return JokerResult(mult_mod=extra.get("mult", 20))
    return None


@register("j_abstract")
def _abstract(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Abstract Joker: +3 mult per joker owned. Source: card.lua:3678."""
    if ctx.joker_main:
        return JokerResult(
            mult_mod=ctx.joker_count * card.ability.get("extra", 3),
        )
    return None


@register("j_acrobat")
def _acrobat(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Acrobat: x3 mult on last hand of round. Source: card.lua:3688."""
    if ctx.joker_main and ctx.hands_left == 0:
        return JokerResult(Xmult_mod=card.ability.get("extra", 3))
    return None


@register("j_mystic_summit")
def _mystic_summit(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Mystic Summit: +15 mult when 0 discards left. Source: card.lua:3694."""
    if ctx.joker_main:
        extra = card.ability.get("extra", {})
        if ctx.discards_left == extra.get("d_remaining", 0):
            return JokerResult(mult_mod=extra.get("mult", 15))
    return None


@register("j_banner")
def _banner(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Banner: +30 chips per discard remaining. Source: card.lua:3707."""
    if ctx.joker_main and ctx.discards_left > 0:
        return JokerResult(
            chip_mod=ctx.discards_left * card.ability.get("extra", 30),
        )
    return None


@register("j_blue_joker")
def _blue_joker(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Blue Joker: +2 chips per card remaining in deck. Source: card.lua:3887."""
    if ctx.joker_main and ctx.deck_cards_remaining > 0:
        return JokerResult(
            chip_mod=card.ability.get("extra", 2) * ctx.deck_cards_remaining,
        )
    return None


@register("j_erosion")
def _erosion(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Erosion: +4 mult per card below starting deck size. Source: card.lua:3894."""
    if ctx.joker_main:
        below = ctx.starting_deck_size - ctx.playing_cards_count
        if below > 0:
            return JokerResult(
                mult_mod=card.ability.get("extra", 4) * below,
            )
    return None


@register("j_stone")
def _stone_joker(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Stone Joker: +25 chips per Stone Card in full deck. Source: card.lua:3922."""
    if ctx.joker_main and ctx.stone_tally > 0:
        return JokerResult(
            chip_mod=card.ability.get("extra", 25) * ctx.stone_tally,
        )
    return None


@register("j_steel_joker")
def _steel_joker(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Steel Joker: x(1 + 0.2 × steel_count) mult. Source: card.lua:3929."""
    if ctx.joker_main and ctx.steel_tally > 0:
        return JokerResult(
            Xmult_mod=1 + card.ability.get("extra", 0.2) * ctx.steel_tally,
        )
    return None


@register("j_bull")
def _bull(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Bull: +2 chips per $1 held. Source: card.lua:3936."""
    if ctx.joker_main and ctx.money > 0:
        return JokerResult(
            chip_mod=card.ability.get("extra", 2) * max(0, ctx.money),
        )
    return None


@register("j_drivers_license")
def _drivers_license(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Driver's License: x3 if ≥16 enhanced cards in deck. Source: card.lua:3943."""
    if ctx.joker_main and ctx.enhanced_card_count >= 16:
        return JokerResult(Xmult_mod=card.ability.get("extra", 3))
    return None


@register("j_blackboard")
def _blackboard(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Blackboard: x3 if ALL held cards are Spades or Clubs. Source: card.lua:3951.

    Uses ``flush_calc=True`` in is_suit calls (matching source).
    """
    if ctx.joker_main and ctx.held_cards is not None:
        if not ctx.held_cards:
            return None
        for c in ctx.held_cards:
            if not (
                c.is_suit("Clubs", flush_calc=True)
                or c.is_suit("Spades", flush_calc=True)
            ):
                return None
        return JokerResult(Xmult_mod=card.ability.get("extra", 3))
    return None


@register("j_stencil")
def _stencil(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Joker Stencil: xMult = empty joker slots. Source: card.lua:3966.

    x_mult is pre-computed as ``joker_slots - joker_count + stencil_count``
    and stored on card.ability.x_mult (source line 4203-4206).
    """
    if ctx.joker_main:
        x = card.ability.get("x_mult", 0)
        if x > 1:
            return JokerResult(Xmult_mod=x)
    return None


@register("j_flower_pot")
def _flower_pot(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Flower Pot: xMult if scoring hand contains all 4 suits. Source: card.lua:3807.

    Non-Wild cards counted first, then Wild cards fill remaining suits.
    """
    if ctx.joker_main and ctx.scoring_hand:
        suits = {"Hearts": 0, "Diamonds": 0, "Spades": 0, "Clubs": 0}
        for c in ctx.scoring_hand:
            if c.ability.get("name") != "Wild Card":
                for s in suits:
                    if suits[s] == 0 and c.is_suit(s, bypass_debuff=True):
                        suits[s] += 1
                        break
        for c in ctx.scoring_hand:
            if c.ability.get("name") == "Wild Card":
                for s in suits:
                    if suits[s] == 0 and c.is_suit(s):
                        suits[s] += 1
                        break
        if all(v > 0 for v in suits.values()):
            return JokerResult(Xmult_mod=card.ability.get("extra", 3))
    return None


@register("j_seeing_double")
def _seeing_double(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Seeing Double: xMult if hand has Club + card of another suit. Source: card.lua:3840.

    Non-Wild cards counted first, then Wild cards fill (Clubs priority).
    """
    if ctx.joker_main and ctx.scoring_hand:
        suits = {"Hearts": 0, "Diamonds": 0, "Spades": 0, "Clubs": 0}
        for c in ctx.scoring_hand:
            if c.ability.get("name") != "Wild Card":
                for s in suits:
                    if c.is_suit(s):
                        suits[s] += 1
        for c in ctx.scoring_hand:
            if c.ability.get("name") == "Wild Card":
                if suits["Clubs"] == 0 and c.is_suit("Clubs"):
                    suits["Clubs"] += 1
                elif suits["Diamonds"] == 0 and c.is_suit("Diamonds"):
                    suits["Diamonds"] += 1
                elif suits["Spades"] == 0 and c.is_suit("Spades"):
                    suits["Spades"] += 1
                elif suits["Hearts"] == 0 and c.is_suit("Hearts"):
                    suits["Hearts"] += 1
        has_club = suits["Clubs"] > 0
        has_other = (
            suits["Hearts"] > 0
            or suits["Diamonds"] > 0
            or suits["Spades"] > 0
        )
        if has_club and has_other:
            return JokerResult(Xmult_mod=card.ability.get("extra", 2))
    return None


@register("j_bootstraps")
def _bootstraps(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Bootstraps: +2 mult per $5 held. Source: card.lua:4046."""
    if ctx.joker_main:
        extra = card.ability.get("extra", {})
        per = extra.get("dollars", 5)
        buckets = ctx.money // per if per > 0 else 0
        if buckets >= 1:
            return JokerResult(mult_mod=extra.get("mult", 2) * buckets)
    return None


@register("j_fortune_teller")
def _fortune_teller(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Fortune Teller: +mult = total tarot uses in run. Source: card.lua:4016."""
    if ctx.joker_main and ctx.consumable_usage_tarot > 0:
        return JokerResult(mult_mod=ctx.consumable_usage_tarot)
    return None


@register("j_loyalty_card")
def _loyalty_card(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Loyalty Card: xMult every N+1 hands. Source: card.lua:3632.

    Formula: ``(every-1-(hands_played-hands_at_create)) % (every+1)``.
    Triggers when result equals ``every``.
    """
    if ctx.joker_main:
        extra = card.ability.get("extra", {})
        every = extra.get("every", 5)
        hands_at_create = card.ability.get("hands_played_at_create", 0)
        remaining = (every - 1 - (ctx.hands_played - hands_at_create)) % (every + 1)
        if remaining == every:
            return JokerResult(Xmult_mod=extra.get("Xmult", 4))
    return None


# ---------------------------------------------------------------------------
# Joker handlers — held-card effects (individual context, cardarea='hand')
# ---------------------------------------------------------------------------

@register("j_raised_fist")
def _raised_fist(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Raised Fist: +2× lowest held card's rank as mult. Source: card.lua:3320.

    Finds the lowest-rank non-Stone card in held_cards, then fires only
    when other_card is that specific card (identity check).
    """
    if ctx.individual and ctx.cardarea == "hand" and ctx.other_card is not None:
        if ctx.held_cards:
            lowest_id = 15
            lowest_card = None
            for c in ctx.held_cards:
                if c.ability.get("effect") != "Stone Card" and c.get_id() < lowest_id:
                    lowest_id = c.get_id()
                    lowest_card = c
            if ctx.other_card is lowest_card:
                if ctx.other_card.debuff:
                    return JokerResult(message="Debuffed")
                nominal = ctx.other_card.base.nominal if ctx.other_card.base else 0
                return JokerResult(h_mult=2 * nominal)
    return None


@register("j_shoot_the_moon")
def _shoot_the_moon(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Shoot the Moon: +13 mult per Queen held. Source: card.lua:3272."""
    if ctx.individual and ctx.cardarea == "hand" and ctx.other_card is not None:
        if ctx.other_card.get_id() == 12:
            if ctx.other_card.debuff:
                return JokerResult(message="Debuffed")
            return JokerResult(h_mult=card.ability.get("extra", 13))
    return None


@register("j_baron")
def _baron(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Baron: x1.5 per King held. Source: card.lua:3287."""
    if ctx.individual and ctx.cardarea == "hand" and ctx.other_card is not None:
        if ctx.other_card.get_id() == 13:
            if ctx.other_card.debuff:
                return JokerResult(message="Debuffed")
            return JokerResult(x_mult=card.ability.get("extra", 1.5))
    return None


# ---------------------------------------------------------------------------
# Joker handlers — scoring-phase economy (individual context)
# ---------------------------------------------------------------------------

@register("j_ticket")
def _golden_ticket(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Golden Ticket: +$4 per Gold Card played. Source: card.lua:3150."""
    if ctx.individual and ctx.cardarea == "play" and ctx.other_card is not None:
        if ctx.other_card.ability.get("name") == "Gold Card":
            return JokerResult(dollars=card.ability.get("extra", 4))
    return None


@register("j_business")
def _business(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Business Card: face card played → 1/2 chance → +$2. Source: card.lua:3175.

    Probabilistic: ``pseudorandom('business') < normal/extra``.
    Returns hardcoded $2 (not from config).
    """
    if ctx.individual and ctx.cardarea == "play" and ctx.other_card is not None:
        if ctx.other_card.is_face(pareidolia=ctx.pareidolia):
            if ctx.rng is not None:
                odds = card.ability.get("extra", 2)
                if ctx.rng.random("business") < ctx.probabilities_normal / odds:
                    return JokerResult(dollars=2)
    return None


@register("j_reserved_parking")
def _reserved_parking(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Reserved Parking: face card held → 1/2 chance → +$1. Source: card.lua:3302.

    Fires in individual/hand context (held cards). Checks debuff.
    """
    if ctx.individual and ctx.cardarea == "hand" and ctx.other_card is not None:
        if ctx.other_card.is_face(pareidolia=ctx.pareidolia):
            extra = card.ability.get("extra", {})
            odds = extra.get("odds", 2)
            if ctx.rng is not None:
                if ctx.rng.random("parking") < ctx.probabilities_normal / odds:
                    if ctx.other_card.debuff:
                        return JokerResult(message="Debuffed")
                    return JokerResult(dollars=extra.get("dollars", 1))
    return None


# ---------------------------------------------------------------------------
# Joker handlers — discard-phase economy (discard context)
# ---------------------------------------------------------------------------

@register("j_faceless")
def _faceless(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Faceless Joker: +$5 if ≥3 face cards discarded. Source: card.lua:2858.

    Fires once per discard when other_card is the last card in full_hand.
    """
    if ctx.discard and ctx.other_card is not None and ctx.full_hand:
        if ctx.other_card is ctx.full_hand[-1]:
            extra = card.ability.get("extra", {})
            face_count = sum(
                1 for c in ctx.full_hand
                if c.is_face(pareidolia=ctx.pareidolia)
            )
            if face_count >= extra.get("faces", 3):
                return JokerResult(dollars=extra.get("dollars", 5))
    return None


@register("j_mail")
def _mail(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Mail-In Rebate: +$5 per card discarded matching mail_card rank.

    Source: card.lua:2825. Fires per discarded card.
    """
    if ctx.discard and ctx.other_card is not None:
        if not ctx.other_card.debuff and ctx.mail_card_id is not None:
            if ctx.other_card.get_id() == ctx.mail_card_id:
                return JokerResult(dollars=card.ability.get("extra", 5))
    return None


@register("j_trading")
def _trading(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Trading Card: first discard of round, single card → +$3, destroy card.

    Source: card.lua:2802. Returns extra={'destroy': True} to signal destruction.
    """
    if ctx.discard and ctx.full_hand is not None:
        if ctx.discards_used <= 0 and len(ctx.full_hand) == 1:
            return JokerResult(
                dollars=card.ability.get("extra", 3),
                remove=True,
                extra={"destroy": True},
            )
    return None


# ---------------------------------------------------------------------------
# Joker handlers — hand-type economy (joker_main context)
# ---------------------------------------------------------------------------

@register("j_to_do_list")
def _to_do_list(card: Card, ctx: JokerContext) -> JokerResult | None:
    """To Do List: +$4 if hand matches to_do_poker_hand. Source: card.lua:3491."""
    if ctx.joker_main and ctx.scoring_name:
        target = card.ability.get("to_do_poker_hand")
        if target and ctx.scoring_name == target:
            extra = card.ability.get("extra", {})
            return JokerResult(dollars=extra.get("dollars", 4))
    return None


# ---------------------------------------------------------------------------
# Joker handlers — boss blind reaction
# ---------------------------------------------------------------------------

@register("j_matador")
def _matador(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Matador: +$8 when boss blind's debuff effect triggers. Source: card.lua:2736."""
    if ctx.debuffed_hand and ctx.blind is not None and ctx.blind.triggered:
        return JokerResult(dollars=card.ability.get("extra", 8))
    return None


# ---------------------------------------------------------------------------
# Joker handlers — copy/delegation (Blueprint, Brainstorm)
# Source: card.lua:2304-2334
# ---------------------------------------------------------------------------

def _find_right_neighbor(card: Card, ctx: JokerContext) -> Card | None:
    """Find the joker immediately to the right of *card* in the joker list."""
    if ctx.jokers is None:
        return None
    for i, j in enumerate(ctx.jokers):
        if j is card and i + 1 < len(ctx.jokers):
            return ctx.jokers[i + 1]
    return None


def _find_leftmost(card: Card, ctx: JokerContext) -> Card | None:
    """Find the leftmost joker that isn't *card*."""
    if ctx.jokers is None:
        return None
    for j in ctx.jokers:
        if j is not card:
            return j
    return None


@register("j_blueprint")
def _blueprint(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Blueprint: copy the joker to the right. Source: card.lua:2304.

    Loop prevention: ``blueprint > len(jokers) + 1`` matches source.
    """
    joker_count = len(ctx.jokers) if ctx.jokers else 0
    bp = (ctx.blueprint + 1) if ctx.blueprint else 1
    if bp > joker_count + 1:
        return None
    target = _find_right_neighbor(card, ctx)
    if target is None or target.debuff:
        return None
    new_ctx = replace(
        ctx,
        blueprint=bp,
        blueprint_card=ctx.blueprint_card or card,
    )
    return calculate_joker(target, new_ctx)


@register("j_brainstorm")
def _brainstorm(card: Card, ctx: JokerContext) -> JokerResult | None:
    """Brainstorm: copy the leftmost joker. Source: card.lua:2321.

    Loop prevention same as Blueprint.
    """
    joker_count = len(ctx.jokers) if ctx.jokers else 0
    bp = (ctx.blueprint + 1) if ctx.blueprint else 1
    if bp > joker_count + 1:
        return None
    target = _find_leftmost(card, ctx)
    if target is None or target.debuff:
        return None
    new_ctx = replace(
        ctx,
        blueprint=bp,
        blueprint_card=ctx.blueprint_card or card,
    )
    return calculate_joker(target, new_ctx)

"""Integration tests: consumable effects chained into scoring/economy.

Each test class exercises a complete workflow:
  consumable use → apply result → score/evaluate → assert outcome.

Effects are applied manually (as the state machine will do in M12) to keep
tests self-contained without requiring the full game loop.
"""

from __future__ import annotations

import pytest

from jackdaw.engine.blind import Blind
from jackdaw.engine.card import Card, reset_sort_id_counter
from jackdaw.engine.consumables import ConsumableContext, ConsumableResult, use_consumable
from jackdaw.engine.economy import calculate_round_earnings
from jackdaw.engine.hand_levels import HandLevels
from jackdaw.engine.rng import PseudoRandom
from jackdaw.engine.scoring import score_hand, score_hand_base
from jackdaw.engine.vouchers import apply_voucher


@pytest.fixture(autouse=True)
def _reset():
    reset_sort_id_counter()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _c(suit_letter: str, rank_letter: str, enh: str = "c_base") -> Card:
    """Create a playing card from single-letter suit/rank codes.

    Suit: H/D/C/S.  Rank: 2-9/T/J/Q/K/A.
    """
    suits = {"H": "Hearts", "D": "Diamonds", "C": "Clubs", "S": "Spades"}
    ranks = {
        "2": "2",
        "3": "3",
        "4": "4",
        "5": "5",
        "6": "6",
        "7": "7",
        "8": "8",
        "9": "9",
        "T": "10",
        "J": "Jack",
        "Q": "Queen",
        "K": "King",
        "A": "Ace",
    }
    suit = suits[suit_letter]
    rank = ranks[rank_letter]
    c = Card()
    c.set_base(f"{suit_letter}_{rank_letter}", suit, rank)
    c.set_ability(enh)
    return c


def _consumable(key: str) -> Card:
    """Create a consumable card from a P_CENTERS key."""
    c = Card()
    c.set_ability(key)
    return c


def _joker(key: str, **kw) -> Card:
    """Create a joker stub using set_ability for proper ability state."""
    j = Card()
    j.set_ability(key)
    j.ability.update(kw)
    return j


def _sb() -> Blind:
    return Blind.create("bl_small", ante=1)


def _bb() -> Blind:
    return Blind.create("bl_big", ante=1)


def _rng(seed: str = "INTEG") -> PseudoRandom:
    return PseudoRandom(seed)


class _ControlledRng:
    """Minimal RNG stub: scripted random values, fixed seed/element."""

    def __init__(self, random_values: list[float]) -> None:
        self._vals = iter(random_values)

    def random(self, key: str) -> float:  # noqa: ARG002
        return next(self._vals)

    def seed(self, key: str) -> float:  # noqa: ARG002
        return 0.5

    def element(self, table: list, seed_val: float) -> tuple:  # noqa: ARG002
        return (table[0], 0)

    def shuffle(self, _lst: list, _seed_val: float) -> None:
        pass


# ---------------------------------------------------------------------------
# Helpers to apply ConsumableResult fields (mirrors the M12 state machine)
# ---------------------------------------------------------------------------


def _apply_result(result: ConsumableResult, hand_levels: HandLevels) -> None:
    """Apply level_up entries from a ConsumableResult to HandLevels."""
    for hand_type, amount in result.level_up or []:
        hand_levels.level_up(hand_type, amount)


def _apply_copy_card(source: Card, target: Card) -> None:
    """Copy rank, suit, enhancement, edition, and seal from source to target.

    Mirrors what the M12 state machine will do with copy_card descriptors.
    """
    if source.base is not None and target.base is not None:
        target.set_base(source.card_key or "", source.base.suit.value, source.base.rank.value)
    target.enhance(source.center_key)
    target.set_edition(source.edition)
    target.set_seal(source.seal)


# ============================================================================
# 1. Chariot → Steel Card → x1.5 held-in-hand mult
# ============================================================================


class TestChariotIntegration:
    """Use The Chariot on an Ace of Spades → becomes Steel Card.
    Score with Ace held in hand → verify x1.5 applies to mult.
    """

    def test_chariot_produces_enhance_steel(self):
        ace = _c("S", "A")
        chariot = _consumable("c_chariot")
        result = use_consumable(
            chariot,
            ConsumableContext(
                card=chariot,
                highlighted=[ace],
            ),
        )
        assert result is not None
        assert result.enhance == [(ace, "m_steel")]

    def test_apply_chariot_and_score_held(self):
        ace = _c("S", "A")
        chariot = _consumable("c_chariot")

        result = use_consumable(
            chariot,
            ConsumableContext(
                card=chariot,
                highlighted=[ace],
            ),
        )
        # Apply the enhancement
        for card, enh_key in result.enhance:
            card.enhance(enh_key)

        # Steel Card h_x_mult = 1.5 (held in hand)
        assert ace.ability.get("h_x_mult") == 1.5

        # Score: play a 2 of Clubs (High Card), hold the Steel Ace
        two = _c("C", "2")
        r = score_hand_base(
            played_cards=[two],
            held_cards=[ace],
            hand_levels=HandLevels(),
            blind=_sb(),
            rng=_rng(),
        )
        # High Card base: 5 chips, 1 mult
        # Phase 7: 2 nominal = +2 chips → 7 chips
        # Phase 8: Steel Ace held → h_x_mult = 1.5 → mult *= 1.5 → 1.5
        # total = floor(7 × 1.5) = 10
        assert r.hand_type == "High Card"
        assert r.mult == 1.5
        assert r.total == 10

    def test_no_x_mult_without_chariot(self):
        """Baseline: plain Ace held gives no x_mult."""
        ace = _c("S", "A")
        two = _c("C", "2")
        r = score_hand_base(
            played_cards=[two],
            held_cards=[ace],
            hand_levels=HandLevels(),
            blind=_sb(),
            rng=_rng(),
        )
        assert r.mult == 1.0
        assert r.total == 7  # floor(7 × 1.0)


# ============================================================================
# 2. Mercury (Planet) → Pair level 2 → leveled chips/mult
# ============================================================================


class TestMercuryIntegration:
    """Use Mercury to level Pair from 1→2.  Score a Pair and verify the
    leveled chips (25) and mult (3) apply.
    """

    def test_mercury_produces_level_up_pair(self):
        mercury = _consumable("c_mercury")
        result = use_consumable(
            mercury,
            ConsumableContext(
                card=mercury,
                game_state={},
            ),
        )
        assert result is not None
        assert ("Pair", 1) in result.level_up

    def test_mercury_then_score_pair(self):
        levels = HandLevels()
        mercury = _consumable("c_mercury")

        result = use_consumable(
            mercury,
            ConsumableContext(
                card=mercury,
                game_state={},
            ),
        )
        _apply_result(result, levels)

        # Pair at level 2: chips = 10 + 15*(2-1) = 25, mult = 2 + 1*(2-1) = 3
        chips, mult = levels.get("Pair")
        assert chips == 25
        assert mult == 3

        # Score Pair of 5s (5+5=10 card chips)
        r = score_hand_base(
            played_cards=[_c("H", "5"), _c("S", "5")],
            held_cards=[],
            hand_levels=levels,
            blind=_sb(),
            rng=_rng(),
        )
        assert r.hand_type == "Pair"
        assert r.chips == 35  # 25 + 5 + 5
        assert r.mult == 3
        assert r.total == 105  # 35 × 3

    def test_pair_without_mercury_baseline(self):
        """Pair at level 1: chips=10, mult=2, total=40."""
        r = score_hand_base(
            played_cards=[_c("H", "5"), _c("S", "5")],
            held_cards=[],
            hand_levels=HandLevels(),
            blind=_sb(),
            rng=_rng(),
        )
        assert r.chips == 20  # 10 + 5 + 5
        assert r.mult == 2
        assert r.total == 40


# ============================================================================
# 3. Black Hole → all 12 hands leveled
# ============================================================================


class TestBlackHoleIntegration:
    """Use Black Hole to level up all 12 hand types.  Verify each is at
    level 2 and that scored hands reflect the upgrade.
    """

    def _use_black_hole(self) -> tuple[ConsumableResult, HandLevels]:
        levels = HandLevels()
        bh = _consumable("c_black_hole")
        result = use_consumable(bh, ConsumableContext(card=bh, game_state={}))
        _apply_result(result, levels)
        return result, levels

    def test_black_hole_levels_all_12(self):
        result, levels = self._use_black_hole()
        assert result is not None
        assert len(result.level_up) == 12

    def test_high_card_leveled(self):
        """High Card level 2: chips=15, mult=2."""
        _, levels = self._use_black_hole()
        chips, mult = levels.get("High Card")
        assert chips == 15  # 5 + 10*(2-1)
        assert mult == 2  # 1 + 1*(2-1)

    def test_pair_leveled(self):
        _, levels = self._use_black_hole()
        chips, mult = levels.get("Pair")
        assert chips == 25  # 10 + 15
        assert mult == 3  # 2 + 1

    def test_full_house_leveled(self):
        _, levels = self._use_black_hole()
        chips, mult = levels.get("Full House")
        assert chips == 65  # 40 + 25
        assert mult == 6  # 4 + 2

    def test_score_high_card_after_black_hole(self):
        """High Card Ace after Black Hole: (15+11) × 2 = 52."""
        _, levels = self._use_black_hole()
        r = score_hand_base(
            played_cards=[_c("S", "A")],
            held_cards=[],
            hand_levels=levels,
            blind=_sb(),
            rng=_rng(),
        )
        assert r.hand_type == "High Card"
        assert r.chips == 26  # 15 + 11 (Ace nominal)
        assert r.mult == 2
        assert r.total == 52

    def test_score_pair_after_black_hole(self):
        _, levels = self._use_black_hole()
        r = score_hand_base(
            played_cards=[_c("H", "5"), _c("S", "5")],
            held_cards=[],
            hand_levels=levels,
            blind=_sb(),
            rng=_rng(),
        )
        assert r.chips == 35  # 25 + 5 + 5
        assert r.mult == 3
        assert r.total == 105


# ============================================================================
# 4. Strength on King → becomes Ace → chip value 11 not 10
# ============================================================================


class TestStrengthIntegration:
    """Use Strength tarot on a King of Spades → rank becomes Ace.
    Score: Ace chip value = 11, not 10.
    """

    def test_strength_produces_change_rank_ace(self):
        king = _c("S", "K")
        strength = _consumable("c_strength")
        result = use_consumable(
            strength,
            ConsumableContext(
                card=strength,
                highlighted=[king],
            ),
        )
        assert result is not None
        # change_rank stores (card, new_rank_string)
        assert len(result.change_rank) == 1
        card, new_rank = result.change_rank[0]
        assert card is king
        assert new_rank == "Ace"

    def test_apply_strength_and_score(self):
        king = _c("S", "K")
        assert king.base.nominal == 10  # King nominal = 10

        strength = _consumable("c_strength")
        result = use_consumable(
            strength,
            ConsumableContext(
                card=strength,
                highlighted=[king],
            ),
        )
        # Apply rank change
        for card, new_rank in result.change_rank:
            card.change_rank(new_rank)

        assert king.base.nominal == 11  # Ace nominal = 11

        # Score a High Card using the upgraded King (now Ace)
        r = score_hand_base(
            played_cards=[king],
            held_cards=[],
            hand_levels=HandLevels(),
            blind=_sb(),
            rng=_rng(),
        )
        assert r.hand_type == "High Card"
        # High Card base=5, Ace nominal=11 → chips=16
        assert r.chips == 16
        assert r.total == 16  # × 1 mult

    def test_king_score_before_strength(self):
        """Baseline: King gives chip value 10."""
        king = _c("S", "K")
        r = score_hand_base(
            played_cards=[king],
            held_cards=[],
            hand_levels=HandLevels(),
            blind=_sb(),
            rng=_rng(),
        )
        assert r.chips == 15  # 5 + 10
        assert r.total == 15


# ============================================================================
# 5. Death: copy Polychrome Glass 7H onto plain 2C
# ============================================================================


class TestDeathIntegration:
    """Death tarot copies the rightmost card onto the other.

    Setup: 2 of Clubs (target, lower sort_id) + 7 of Hearts with Polychrome
    edition and Glass enhancement (source, higher sort_id).
    After Death: 2 of Clubs becomes a Polychrome Glass 7 of Hearts.
    """

    def test_death_copy_card_descriptor(self):
        target = _c("C", "2")  # created first → lower sort_id
        source = _c("H", "7", "m_glass")  # created second → higher sort_id
        source.set_edition({"polychrome": True})

        death = _consumable("c_death")
        result = use_consumable(
            death,
            ConsumableContext(
                card=death,
                highlighted=[target, source],
            ),
        )
        assert result is not None
        assert result.copy_card is not None
        copied_source, copied_target = result.copy_card
        assert copied_source is source
        assert copied_target is target

    def test_apply_death_verifies_rank_suit_edition_enhancement(self):
        target = _c("C", "2")
        source = _c("H", "7", "m_glass")
        source.set_edition({"polychrome": True})

        death = _consumable("c_death")
        result = use_consumable(
            death,
            ConsumableContext(
                card=death,
                highlighted=[target, source],
            ),
        )
        # Apply the copy
        src, tgt = result.copy_card
        _apply_copy_card(src, tgt)

        # Target is now a 7 of Hearts
        assert target.base.rank.value == "7"
        assert target.base.suit.value == "Hearts"
        # Enhancement: Glass Card
        assert target.ability.get("effect") == "Glass Card"
        # Edition: Polychrome
        assert target.edition is not None
        assert target.edition.get("polychrome") is True

    def test_apply_death_score_verifies_glass_x_mult(self):
        """After copy, scoring the 7H card as Glass should give x2 mult."""
        target = _c("C", "2")
        source = _c("H", "7", "m_glass")

        death = _consumable("c_death")
        result = use_consumable(
            death,
            ConsumableContext(
                card=death,
                highlighted=[target, source],
            ),
        )
        _apply_copy_card(*result.copy_card)

        # Score the updated target (now Glass 7H): Glass x_mult = 2.0
        r = score_hand_base(
            played_cards=[target],
            held_cards=[],
            hand_levels=HandLevels(),
            blind=_sb(),
            rng=_rng(),
        )
        # High Card: 5 base + 7 nominal = 12 chips; Glass x2 → mult *= 2 → 2
        assert r.hand_type == "High Card"
        assert r.chips == 12
        assert r.mult == 2.0
        assert r.total == 24


# ============================================================================
# 6. Hanged Man on 2 cards → destroyed, deck has 50 cards
# ============================================================================


class TestHangedManIntegration:
    """Use Hanged Man on 2 cards → consume destroys them.
    Deck (playing_cards) drops from 52 to 50.
    """

    def _make_deck(self) -> list[Card]:
        suits = ["H", "D", "C", "S"]
        ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
        deck = []
        for s in suits:
            for r in ranks:
                deck.append(_c(s, r))
        return deck  # 52 cards

    def test_hanged_man_produces_destroy(self):
        deck = self._make_deck()
        card1, card2 = deck[0], deck[1]
        hanged = _consumable("c_hanged_man")
        result = use_consumable(
            hanged,
            ConsumableContext(
                card=hanged,
                highlighted=[card1, card2],
                playing_cards=deck,
            ),
        )
        assert result is not None
        assert result.destroy == [card1, card2]

    def test_apply_hanged_man_deck_shrinks_to_50(self):
        deck = self._make_deck()
        assert len(deck) == 52

        card1, card2 = deck[5], deck[10]
        hanged = _consumable("c_hanged_man")
        result = use_consumable(
            hanged,
            ConsumableContext(
                card=hanged,
                highlighted=[card1, card2],
                playing_cards=deck,
            ),
        )
        # Apply: remove destroyed cards from playing_cards
        for card in result.destroy:
            deck.remove(card)

        assert len(deck) == 50
        assert card1 not in deck
        assert card2 not in deck

    def test_hanged_man_single_card(self):
        """Hanged Man with 1 highlighted card → deck shrinks to 51."""
        deck = self._make_deck()
        card = deck[3]
        hanged = _consumable("c_hanged_man")
        result = use_consumable(
            hanged,
            ConsumableContext(
                card=hanged,
                highlighted=[card],
                playing_cards=deck,
            ),
        )
        for c in result.destroy:
            deck.remove(c)
        assert len(deck) == 51


# ============================================================================
# 7. Wheel of Fortune (mock RNG success) → joker gets edition → score
# ============================================================================


class TestWheelOfFortuneIntegration:
    """WoF with RNG scripted for success → joker gains Foil edition.
    Score a High Card with that joker → verify Foil's +50 chip_mod applies.
    """

    def test_wheel_of_fortune_produces_foil(self):
        """Roll 0.1 (< 0.25 → success), poll 0.3 (≤ 0.5 → foil)."""
        j = _joker("j_joker")
        wof = _consumable("c_wheel_of_fortune")
        rng = _ControlledRng([0.1, 0.3])
        result = use_consumable(
            wof,
            ConsumableContext(
                card=wof,
                jokers=[j],
                rng=rng,
                game_state={"probabilities_normal": 1},
            ),
        )
        assert result is not None
        assert result.add_edition is not None
        assert result.add_edition["target"] is j
        assert result.add_edition["edition"] == {"foil": True}

    def test_apply_foil_and_score(self):
        """Foil joker adds +50 chip_mod in Phase 9a.

        High Card Ace (5+11=16 chips) + j_joker (+4 mult) + Foil (+50 chips):
          chips = 16 + 50 = 66, mult = 1+4 = 5, total = 330
        """
        j = _joker("j_joker")
        wof = _consumable("c_wheel_of_fortune")
        rng = _ControlledRng([0.1, 0.3])
        result = use_consumable(
            wof,
            ConsumableContext(
                card=wof,
                jokers=[j],
                rng=rng,
                game_state={"probabilities_normal": 1},
            ),
        )
        # Apply the edition to the joker
        j.set_edition(result.add_edition["edition"])
        assert j.edition is not None and j.edition.get("foil") is True

        # Score High Card Ace with the foil joker
        ace = _c("S", "A")
        r = score_hand(
            played_cards=[ace],
            held_cards=[],
            jokers=[j],
            hand_levels=HandLevels(),
            blind=_sb(),
            rng=_rng(),
        )
        # High Card 5 + Ace 11 = 16 chips
        # Phase 9a: foil chip_mod +50 → 66 chips
        # Phase 9b: j_joker +4 mult → 5 mult
        assert r.chips == 66
        assert r.mult == 5.0
        assert r.total == 330

    def test_foil_vs_no_foil_difference(self):
        """Foil adds +50 chips, changing total from 80 to 330."""
        ace = _c("S", "A")

        # Without foil
        j_plain = _joker("j_joker")
        r_plain = score_hand(
            played_cards=[ace],
            held_cards=[],
            jokers=[j_plain],
            hand_levels=HandLevels(),
            blind=_sb(),
            rng=_rng(),
        )
        # 16 chips × (1+4) mult = 80
        assert r_plain.total == 80

        # With foil
        ace2 = _c("S", "A")
        j_foil = _joker("j_joker")
        j_foil.set_edition({"foil": True})
        r_foil = score_hand(
            played_cards=[ace2],
            held_cards=[],
            jokers=[j_foil],
            hand_levels=HandLevels(),
            blind=_sb(),
            rng=_rng(),
        )
        # (16+50) chips × 5 mult = 330
        assert r_foil.total == 330
        assert r_foil.total - r_plain.total == 250  # 50 chips × 5 mult


# ============================================================================
# 8. v_grabber + v_nacho_tong → +2 hands per round
# ============================================================================


class TestVoucherHandsIntegration:
    """Apply Grabber then Nacho Tong → round_resets.hands goes from 4 to 6."""

    def test_grabber_plus_nacho_tong(self):
        game_state: dict = {}

        apply_voucher("v_grabber", game_state)
        assert game_state["round_resets"]["hands"] == 5  # 4 + 1

        apply_voucher("v_nacho_tong", game_state)
        assert game_state["round_resets"]["hands"] == 6  # 5 + 1

    def test_grabber_mutations_dict(self):
        mutations = apply_voucher("v_grabber", {})
        assert mutations["round_resets.hands"] == 5

    def test_nacho_tong_alone(self):
        game_state: dict = {"round_resets": {"hands": 4}}
        apply_voucher("v_nacho_tong", game_state)
        assert game_state["round_resets"]["hands"] == 5

    def test_order_independent(self):
        """Applying Nacho Tong then Grabber gives same result."""
        gs1: dict = {}
        apply_voucher("v_grabber", gs1)
        apply_voucher("v_nacho_tong", gs1)

        gs2: dict = {}
        apply_voucher("v_nacho_tong", gs2)
        apply_voucher("v_grabber", gs2)

        assert gs1["round_resets"]["hands"] == gs2["round_resets"]["hands"] == 6


# ============================================================================
# 9. v_clearance_sale → all shop card costs reduced by 25%
# ============================================================================


class TestClearanceSaleIntegration:
    """Apply Clearance Sale voucher → discount_percent=25.
    Verify set_cost() with that discount reduces card costs by 25%.
    """

    def test_clearance_sale_sets_discount(self):
        game_state: dict = {}
        mutations = apply_voucher("v_clearance_sale", game_state)
        assert game_state["discount_percent"] == 25
        assert mutations["discount_percent"] == 25

    def test_joker_cost_discounted(self):
        """j_greedy_joker base_cost=5: floor((5+0.5)*75/100) = floor(4.125) = 4."""
        game_state: dict = {}
        apply_voucher("v_clearance_sale", game_state)

        j = Card()
        j.set_ability("j_greedy_joker")
        j.set_cost(discount_percent=game_state["discount_percent"])
        assert j.cost == 4

    def test_multiple_cards_all_discounted(self):
        """Discount applies to any card — joker, consumable, and voucher."""
        game_state: dict = {}
        apply_voucher("v_clearance_sale", game_state)
        discount = game_state["discount_percent"]

        import math

        test_cases = [
            ("j_joker", 2),  # base_cost=2 → floor(2.5*0.75)=1
            ("j_greedy_joker", 5),  # base_cost=5 → floor(5.5*0.75)=4
        ]
        for key, base_cost in test_cases:
            c = Card()
            c.set_ability(key)
            c.set_cost(discount_percent=discount)
            expected = max(1, math.floor((base_cost + 0.5) * (100 - discount) / 100))
            assert c.cost == expected, f"{key}: expected {expected}, got {c.cost}"

    def test_liquidation_discount_50_percent(self):
        """Liquidation (v_liquidation) sets discount_percent=50."""
        game_state: dict = {}
        apply_voucher("v_liquidation", game_state)

        j = Card()
        j.set_ability("j_greedy_joker")  # base_cost=5
        j.set_cost(discount_percent=game_state["discount_percent"])
        # floor((5+0.5)*50/100) = floor(2.75) = 2
        assert j.cost == 2


# ============================================================================
# 10. Full round earnings: boss blind + Golden Joker + Cloud 9 + rental @ $23
# ============================================================================


class TestFullRoundEarnings:
    """Beat Big Blind with Golden Joker ($4) + Cloud 9 (3 nines, $3) + 1 rental
    joker, 2 hands left, $23 in bank.

    Step-by-step:
      rental_cost     = $3   (1 rental joker × $3/round)
      effective_money = $23 - $3 = $20
      blind_reward    = $4   (Big Blind at ante 1)
      hands_bonus     = $2   (2 × $1/hand)
      discards_bonus  = $0   (no green deck)
      joker_dollars   = $7   (Golden $4 + Cloud 9 $3)
      interest        = $4   (min(20//5=4, 25//5=5) × $1)
      total           = 4 + 2 + 0 + 7 + 4 - 3 = $14
    """

    def _make_jokers(self) -> list[Card]:
        golden = Card()
        golden.center_key = "j_golden"
        golden.ability = {"name": "j_golden", "set": "Joker", "extra": 4}
        golden.sell_cost = 1

        cloud9 = Card()
        cloud9.center_key = "j_cloud_9"
        cloud9.ability = {"name": "j_cloud_9", "set": "Joker", "extra": 1, "nine_tally": 3}
        cloud9.sell_cost = 1

        rental = Card()
        rental.center_key = "j_spare_trousers"
        rental.ability = {"name": "j_spare_trousers", "set": "Joker", "rental": True}
        rental.sell_cost = 1

        return [golden, cloud9, rental]

    def test_full_breakdown(self):
        jokers = self._make_jokers()
        blind = _bb()  # Big Blind: $4 reward
        result = calculate_round_earnings(
            blind=blind,
            hands_left=2,
            discards_left=0,
            money=23,
            jokers=jokers,
            game_state={},
        )
        assert result.blind_reward == 4
        assert result.unused_hands_bonus == 2
        assert result.unused_discards_bonus == 0
        assert result.joker_dollars == 7  # Golden $4 + Cloud 9 $3
        assert result.rental_cost == 3
        assert result.interest == 4  # min(20//5=4, 5) = 4
        assert result.total == 14

    def test_rental_before_interest(self):
        """Rental deducted first: effective_money=20, not 23."""
        jokers = self._make_jokers()
        result = calculate_round_earnings(
            blind=_bb(),
            hands_left=0,
            discards_left=0,
            money=23,
            jokers=jokers,
            game_state={},
        )
        # effective_money = 23 - 3 = 20 → interest = min(4, 5) = 4
        assert result.interest == 4

    def test_no_rental_higher_interest(self):
        """Without rental, effective_money=23, interest = min(4, 5) = 4."""
        jokers = self._make_jokers()[:2]  # Golden + Cloud 9 only (no rental)
        result = calculate_round_earnings(
            blind=_bb(),
            hands_left=0,
            discards_left=0,
            money=23,
            jokers=jokers,
            game_state={},
        )
        assert result.rental_cost == 0
        assert result.interest == 4  # 23//5=4 same bracket as 20//5=4 here

    def test_total_with_no_interest(self):
        """no_interest modifier: total = blind + hands + jokers - rental."""
        jokers = self._make_jokers()
        result = calculate_round_earnings(
            blind=_bb(),
            hands_left=2,
            discards_left=0,
            money=23,
            jokers=jokers,
            game_state={"modifiers": {"no_interest": True}},
        )
        assert result.interest == 0
        assert result.total == 4 + 2 + 7 - 3  # = 10

"""Joker batch D fidelity: the round lifecycle (G55, G97-G106).

Each test cites the vanilla 1.0.1o Lua it pins (``balatro_source/``).
"""

from __future__ import annotations

from jackdaw.engine.actions import GamePhase, SelectBlind, SkipBlind, SkipPack
from jackdaw.engine.game import _award_tag, step
from jackdaw.engine.rng import PseudoRandom
from tests.engine.audit._boards import force_boss, give_consumable, give_joker, ids, start_boss

from ._joker_helpers import discard, give_card, new_run, play, select_blind, win_round


def _stones(gs: dict) -> list:
    owned = gs["deck"] + gs["hand"] + gs["discard_pile"]
    return [c for c in owned if c.center_key == "m_stone"]


def _three_aajk_shuffles(seed: str, cards: list) -> list:
    """What three ``G.jokers:shuffle('aajk')`` calls leave, from a fresh stream."""
    ref = PseudoRandom(seed)
    order = list(cards)
    for _ in range(3):
        ref.shuffle(order, ref.seed("aajk"))
    return order


class TestMarbleStoneIsShuffledIn:
    """G55: Marble Joker queues its events during new_round's setting_blind
    loop: one makes the Stone in G.play, then draw_card moves it to G.deck
    (card.lua:2580-2601).  new_round queues its shuffle event after that loop
    (state_events.lua:335-349), so the round's 'nr' shuffle includes the
    Stone.  (Steamodded queues the draw inside the creation event, after the
    shuffle; the live game deals a stone-less shuffle, smods-diffs.md S01.)"""

    def test_the_round_shuffle_includes_the_stone(self):
        seed = "AUDIT_G55"
        gs = new_run(seed)
        give_joker(gs, "j_marble")
        before = list(gs["deck"])

        step(gs, SelectBlind())

        stones = _stones(gs)
        assert len(stones) == 1
        # pseudoshuffle sorts by sort_id, then shuffles all 53 cards.
        ref = before + stones
        twin = PseudoRandom(seed)
        twin.shuffle(ref, twin.seed("nr1"))
        assert ids(gs["deck"]) == ids(ref[:-8])
        assert sorted(ids(gs["hand"])) == sorted(ids(ref[-8:]))
        assert not gs.get("pending_deck_bottom")

    def test_each_copy_adds_a_stone(self):
        """The branch has no ``context.blueprint`` guard: Blueprint's copy
        makes a second Stone, shuffled in as well."""
        gs = new_run("AUDIT_G55_BP")
        give_joker(gs, "j_blueprint")
        give_joker(gs, "j_marble")
        step(gs, SelectBlind())
        assert len(_stones(gs)) == 2
        assert len(gs["deck"]) + len(gs["hand"]) == 54


class TestBossSetTimeEffectsComeFirst:
    """G97: new_round calls set_blind before the jokers' setting_blind
    context (state_events.lua:333-337).  The Water counts the round's discards
    there (blind.lua:179-182) and its ease_discard is an event queued ahead of
    every joker's, so Burglar's event (card.lua:2522-2528) finds no discards
    left, and a disable later gives back the whole count (blind.lua:361-363)."""

    def test_the_water_counts_its_discards_before_burglar(self):
        gs = new_run("AUDIT_G97")
        give_joker(gs, "j_burglar")
        start_boss(gs, "bl_water")
        rr, cr = gs["round_resets"], gs["current_round"]
        assert gs["blind"].discards_sub == rr["discards"]
        assert cr["discards_left"] == 0
        assert cr["hands_left"] == rr["hands"] + 3

    def test_chicot_gives_the_waters_discards_back_despite_burglar(self):
        """Chicot's disable runs from a nested event (card.lua:2492-2500),
        after Burglar's: The Water's count comes back either way."""
        for board in (("j_burglar", "j_chicot"), ("j_chicot", "j_burglar")):
            gs = new_run("AUDIT_G97C")
            for key in board:
                give_joker(gs, key)
            start_boss(gs, "bl_water")
            assert gs["blind"].disabled, board
            assert gs["current_round"]["discards_left"] == gs["round_resets"]["discards"], board


class TestTheArmCheckTrigger:
    """G98: highlighting a card runs CardArea:parse_highlighted
    (cardarea.lua:156-158, 162-185), whose check-mode debuff_hand sets The
    Arm's ``triggered`` for a hand above level 1, without the demote
    (blind.lua:550-559).  After a discard the flag stays as that check left
    it.  Burnt Joker's levels made it visible."""

    def test_a_levelled_hand_sets_triggered(self):
        gs = new_run("AUDIT_G98")
        start_boss(gs, "bl_arm")
        gs["hand_levels"].level_up("Pair")
        discard(gs, give_card(gs, "S_9"), give_card(gs, "H_9"))
        assert gs["blind"].triggered
        assert gs["hand_levels"]["Pair"].level == 2

    def test_a_level_one_hand_leaves_it_clear(self):
        gs = new_run("AUDIT_G98L")
        start_boss(gs, "bl_arm")
        discard(gs, give_card(gs, "S_9"), give_card(gs, "H_9"))
        assert not gs["blind"].triggered


class TestRoundEndRepricing:
    """G99: Egg and Gift Card end the round with ``set_cost`` (card.lua:2985-
    2988, 2993-3010).  Lua never clears ``ability.couponed``, which zeroes a
    cost only in a shop area (card.lua:383), and set_cost reads Astronomer
    for a held Planet (card.lua:380)."""

    def test_egg_keeps_the_coupon_flag(self):
        gs = select_blind(new_run("AUDIT_G99E"))
        egg = give_joker(gs, "j_egg")
        egg.ability["couponed"] = True
        egg.cost = 0
        win_round(gs)
        assert egg.ability.get("couponed") is True
        assert (egg.cost, egg.sell_cost) == (4, 2 + 3)

    def test_gift_card_keeps_the_coupon_flag_and_reads_astronomer(self):
        gs = select_blind(new_run("AUDIT_G99G"))
        give_joker(gs, "j_gift")
        joker = give_joker(gs, "j_joker")
        joker.ability["couponed"] = True
        give_joker(gs, "j_astronomer")
        pluto = give_consumable(gs, "c_pluto")
        win_round(gs)
        assert joker.ability.get("couponed") is True
        # Pluto: cost 0 with Astronomer; sell max(1, 0) + Gift Card's $1.
        assert (pluto.cost, pluto.sell_cost) == (0, 2)


class TestSettingBlindEventOrder:
    """G100: the jokers' setting_blind events run in board order; the events
    those queue run after them all (state_events.lua:335-349)."""

    def test_cartomancers_tarot_is_made_after_the_stone(self):
        """Cartomancer's event queues the event that makes the Tarot
        (card.lua:2547-2559); Marble's Stone is made by an event of its own
        (card.lua:2581-2590), so it takes the earlier sort id even when
        Cartomancer comes first."""
        gs = new_run("AUDIT_G100C")
        give_joker(gs, "j_cartomancer")
        give_joker(gs, "j_marble")
        select_blind(gs)
        (stone,) = _stones(gs)
        (tarot,) = gs["consumables"]
        assert stone.sort_id < tarot.sort_id

    def test_cartomancers_tarot_is_made_after_riff_raffs_jokers(self):
        gs = new_run("AUDIT_G100R")
        give_joker(gs, "j_cartomancer")
        riff = give_joker(gs, "j_riff_raff")
        select_blind(gs)
        made = [j for j in gs["jokers"] if j.sort_id > riff.sort_id]
        assert len(made) == 2
        assert max(ids(made)) < gs["consumables"][0].sort_id

    def test_amber_acorn_shuffles_the_sliced_joker_too(self):
        """Amber Acorn's shuffles are the first events that events queue
        (blind.lua:195-204); the Dagger's victim leaves by a later one
        (start_dissolve from its event, card.lua:2570-2576), so the shuffles
        still see it."""
        seed = "AUDIT_G100A1"
        gs = new_run(seed)
        keys = ("j_ceremonial", "j_joker", "j_greedy_joker", "j_lusty_joker", "j_wrathful_joker")
        board = [give_joker(gs, k) for k in keys]
        victim = board[1]
        force_boss(gs, "bl_final_acorn")
        step(gs, SelectBlind())
        expected = [j for j in _three_aajk_shuffles(seed, board) if j is not victim]
        assert ids(gs["jokers"]) == ids(expected)
        # (Shuffling the board without the victim would order it differently.)
        rest = [j for j in board if j is not victim]
        assert ids(expected) != ids(_three_aajk_shuffles(seed, rest))

    def test_a_new_joker_redebuffs_the_stone(self):
        """Riff-raff's new jokers queue set_blind(nil, true) from add_to_deck
        (card.lua:641); it runs after Marble's Stone is made, whichever comes
        first on the board, and Verdant Leaf debuffs the Stone
        (blind.lua:207-210, 640-643)."""
        for board in (("j_riff_raff", "j_marble"), ("j_marble", "j_riff_raff")):
            gs = new_run("AUDIT_G100V")
            for key in board:
                give_joker(gs, key)
            start_boss(gs, "bl_final_leaf")
            (stone,) = _stones(gs)
            assert stone.debuff, board

    def test_a_lone_stone_keeps_its_state(self):
        """With nothing added or removed no re-debuff follows the Stone:
        set_blind's own pass ran before it existed (blind.lua:207-210)."""
        gs = new_run("AUDIT_G100L")
        give_joker(gs, "j_marble")
        start_boss(gs, "bl_final_leaf")
        (stone,) = _stones(gs)
        assert not stone.debuff


class TestSlicedHandSize:
    """G105: a sliced joker leaves by its removal event, queued by the slicing
    joker's event (card.lua:2513-2516, 2570-2576), and remove_from_deck's
    G.hand:change_size is one event deeper (card.lua:648-650, cardarea.lua:
    94-111).  The round's deal counts its cards first (game.lua:3219-3222,
    state_events.lua:362), and the change runs while the hand is still
    empty: it neither shrinks the deal nor draws."""

    def test_a_sliced_juggler_still_counts_for_the_deal(self):
        gs = new_run("AUDIT_G105")
        give_joker(gs, "j_ceremonial")
        give_joker(gs, "j_juggler")
        size = gs["hand_size"]
        select_blind(gs)
        assert len(gs["hand"]) == size
        assert gs["hand_size"] == size - 1

    def test_a_sliced_merry_andy_gives_its_card_back_later(self):
        gs = new_run("AUDIT_G105M")
        give_joker(gs, "j_ceremonial")
        give_joker(gs, "j_merry_andy")
        size = gs["hand_size"]
        select_blind(gs)
        assert len(gs["hand"]) == size
        assert gs["hand_size"] == size + 1


class TestMrBonesDissolvesLater:
    """G101: Mr. Bones' save queues the event that dissolves it
    (card.lua:3047-3063).  end_round's loop still charges its rent and ticks
    its sticker (state_events.lua:99-110, card.lua:2271-2289)."""

    def test_a_rental_mr_bones_pays_its_rent_when_it_saves(self):
        gs = select_blind(new_run("AUDIT_G101"))
        bones = give_joker(gs, "j_mr_bones", rental=True)
        gs["current_round"]["hands_left"] = 1
        gs["chips"] = gs["blind"].chips / 2
        dollars = gs["dollars"]
        play(gs, gs["hand"][0])
        assert gs["phase"] == GamePhase.ROUND_EVAL
        assert bones not in gs["jokers"]
        assert gs["dollars"] == dollars - gs["rental_rate"]


class TestCreatedPlayingCardsArePriced:
    """G102: Card:init ends with set_cost (card.lua:72): a playing card costs
    max(1, floor(1.5)) = 1 and sells for 1 (card.lua:369-382)."""

    def test_certificates_card(self):
        """create_playing_card (common_events.lua:1927-1942); set_seal prices
        it again (card.lua:498)."""
        gs = new_run("AUDIT_G102C")
        give_joker(gs, "j_certificate")
        select_blind(gs)
        card = max(gs["hand"], key=lambda c: c.sort_id)
        assert card.seal is not None
        assert (card.cost, card.sell_cost) == (1, 1)

    def test_marbles_stone(self):
        """Marble builds the Stone with Card() (card.lua:2585)."""
        gs = new_run("AUDIT_G102M")
        give_joker(gs, "j_marble")
        select_blind(gs)
        (stone,) = _stones(gs)
        assert (stone.cost, stone.sell_cost) == (1, 1)


def _held(gs: dict, key: str) -> int:
    """Tags of ``key`` still waiting for a new_blind_choice pass."""
    return sum(1 for e in gs["awarded_tags"] if e["key"] == key and not e.get("nbc_fired"))


class TestPackTagsOpenOneAtATime:
    """G106: every new_blind_choice pass over G.GAME.tags stops at the first
    tag that fires (`if ... then break end`: skip_blind, end_consumeable,
    reroll_boss, update_blind_select).  Skipping with two Double Tags held
    makes three Meteor Tags (tag.lua:315-331): the first opens its pack, the
    copies stay held, and each pack's end_consumeable pass opens the next
    (button_callbacks.lua:2617-2619, 2775-2777)."""

    def test_the_copies_wait_for_each_pack_to_close(self):
        gs = new_run("AUDIT_G106")
        _award_tag(gs, "tag_double", source="test")
        _award_tag(gs, "tag_double", source="test")
        gs["round_resets"]["blind_tags"]["Small"] = "tag_meteor"

        step(gs, SkipBlind())
        assert gs["phase"] == GamePhase.PACK_OPENING
        assert _held(gs, "tag_meteor") == 2
        assert not gs.get("pending_tag_packs")

        for held in (1, 0):
            step(gs, SkipPack())
            assert gs["phase"] == GamePhase.PACK_OPENING
            assert _held(gs, "tag_meteor") == held
            assert not gs.get("pending_tag_packs")

        step(gs, SkipPack())
        assert gs["phase"] == GamePhase.BLIND_SELECT
        assert gs["blind_on_deck"] == "Big"

"""Regression tests for skip-tag hook wiring and the pack joker-cap gate.

Both bugs were found by LLM agents playing full runs (2026-07-20):
- Deferred skip tags (store_joker_modify / eval / shop_start /
  shop_final_pass) were awarded but never fired: populate_shop's "M11 tag
  system" note deferred the hooks and nothing wired them.
- Pack picks ignored the joker-slot cap (vanilla button_callbacks.lua:2112
  gates non-negative Jokers when the board is full).
"""
from __future__ import annotations

from jackdaw.engine.actions import GamePhase
from jackdaw.engine.game import _fire_shop_tags, step
from jackdaw.engine.run_init import initialize_run
from jackdaw.env.action_space import ActionType, get_action_mask


def _cheat_through_blind(gs) -> None:
    from jackdaw.engine.actions import CashOut, PlayHand, SelectBlind

    step(gs, SelectBlind())
    gs["chips"] = 0
    gs["current_round"]["hands_left"] = 4
    # Force an instant clear by inflating chips post-selection.
    gs["blind"].chips = 1
    step(gs, PlayHand(card_indices=[0, 1, 2, 3, 4]))
    step(gs, CashOut())


class TestPackJokerCapGate:
    def test_full_board_blocks_joker_picks(self):
        gs = initialize_run("b_red", 1, "OVERFLOW1")
        gs["phase"] = GamePhase.BLIND_SELECT
        gs["blind_on_deck"] = "Small"
        from jackdaw.engine.card_factory import create_joker

        for key in ("j_joker", "j_greedy_joker", "j_lusty_joker",
                    "j_wrathful_joker", "j_gluttenous_joker"):
            gs["jokers"].append(create_joker(key))
        assert len(gs["jokers"]) == 5

        # Fabricate an open buffoon pack.
        gs["phase"] = GamePhase.PACK_OPENING
        gs["pack_type"] = "Buffoon"
        gs["pack_choices_remaining"] = 1
        gs["pack_cards"] = [
            create_joker("j_fortune_teller"),
            create_joker("j_walkie_talkie"),
        ]
        mask = get_action_mask(gs)
        assert not mask.type_mask[ActionType.PickPackCard]

        # Free a slot: picks become legal again.
        gs["jokers"].pop()
        mask = get_action_mask(gs)
        assert mask.type_mask[ActionType.PickPackCard]
        assert mask.entity_masks[ActionType.PickPackCard].all()


class TestSkipTagWiring:
    def test_polychrome_tag_fires_on_next_shop_joker(self):
        # CLAUDEA2: ante-1 Big skip tag is tag_polychrome.
        gs = initialize_run("b_red", 1, "CLAUDEA2")
        gs["phase"] = GamePhase.BLIND_SELECT
        gs["blind_on_deck"] = "Small"
        assert gs["round_resets"]["blind_tags"]["Big"] == "tag_polychrome"

        _cheat_through_blind(gs)  # beat Small, now in shop
        from jackdaw.engine.actions import NextRound, SkipBlind

        step(gs, NextRound())
        step(gs, SkipBlind())  # Big skipped -> polychrome tag awarded
        _cheat_through_blind(gs)  # beat Boss -> next shop generated

        jokers = [c for c in gs["shop_cards"]
                  if (getattr(c, "ability", None) or {}).get("set") == "Joker"]
        assert jokers, "expected a joker in the post-skip shop"
        tagged = [c for c in jokers if (getattr(c, "edition", None) or {}).get("polychrome")]
        assert tagged, "polychrome tag should mark the next base shop joker"
        assert tagged[0].cost == 0  # vanilla: tag-edition jokers are free

    def test_shop_tags_do_not_refire(self):
        gs = initialize_run("b_red", 1, "CLAUDEA2")
        gs["awarded_tags"] = [{"key": "tag_polychrome", "blind": "Big"}]
        from jackdaw.engine.card_factory import create_joker

        gs["shop_cards"] = [create_joker("j_joker")]
        _fire_shop_tags(gs, rerolled=False)
        assert gs["awarded_tags"][0]["shop_fired"]
        gs["shop_cards"] = [create_joker("j_sly")]
        _fire_shop_tags(gs, rerolled=False)
        assert not (getattr(gs["shop_cards"][0], "edition", None) or {})


class TestDoubleTag:
    """Double Tag converts into a copy of the next tag acquired
    (tag.lua 'tag_added'); duplicated pack tags queue their packs."""

    def _skip_into(self, seed: str, n_skips: int):
        from jackdaw.engine.actions import GamePhase, SkipBlind
        from jackdaw.engine.game import step
        from jackdaw.engine.run_init import initialize_run

        gs = initialize_run("b_red", 1, seed)
        gs["phase"] = GamePhase.BLIND_SELECT
        gs["blind_on_deck"] = "Small"
        for _ in range(n_skips):
            step(gs, SkipBlind())
        return gs

    def test_double_tag_duplicates_next_tag(self):
        from jackdaw.engine.game import _check_double_tag

        gs = {"awarded_tags": [{"key": "tag_double", "result": None}], "rng": None}
        _check_double_tag(gs, "tag_economy")
        keys = [e["key"] for e in gs["awarded_tags"]]
        assert keys.count("tag_double") == 0  # consumed
        assert keys.count("tag_economy") == 1  # the duplicate entry

    def test_double_never_duplicates_double(self):
        from jackdaw.engine.game import _check_double_tag

        gs = {"awarded_tags": [{"key": "tag_double", "result": None}], "rng": None}
        _check_double_tag(gs, "tag_double")
        keys = [e["key"] for e in gs["awarded_tags"]]
        assert keys == ["tag_double"]

    def test_second_tag_pack_queues_and_opens_after_close(self):
        from jackdaw.engine.actions import GamePhase
        from jackdaw.engine.game import _close_pack, _open_tag_pack
        from jackdaw.engine.run_init import initialize_run

        gs = initialize_run("b_red", 1, "DBLPACK1")
        gs["phase"] = GamePhase.BLIND_SELECT
        _open_tag_pack(gs, "p_buffoon_mega_1")
        assert gs["phase"] == GamePhase.PACK_OPENING
        first_pack = [c.center_key for c in gs["pack_cards"]]
        _open_tag_pack(gs, "p_buffoon_mega_1")  # queued, not overwritten
        assert [c.center_key for c in gs["pack_cards"]] == first_pack
        assert gs.get("pending_tag_packs") == ["p_buffoon_mega_1"]
        _close_pack(gs)
        assert gs["phase"] == GamePhase.PACK_OPENING  # second pack now open
        assert gs["pack_cards"], "second pack should have cards"
        assert not gs.get("pending_tag_packs")
        _close_pack(gs)
        assert gs["phase"] == GamePhase.BLIND_SELECT

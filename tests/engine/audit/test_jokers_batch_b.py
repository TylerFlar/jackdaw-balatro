"""Joker batch B fidelity: Joker Stencil (G58), set_ability on a card in play (G63) and The
Arm's preview (G64)."""

from __future__ import annotations

from ._helpers import force_boss
from ._joker_helpers import discard, give_card, give_joker, new_run, play, select_blind


class TestStencilSharedRule:
    """G58: Card:update keeps a Stencil's x_mult at the empty joker slots plus one per Stencil
    in G.jokers (card.lua:4203-4208), and the main step's shared rule scores any joker whose
    x_mult is above 1 (card.lua:3653-3659) before the Stencil's own branch, which needs an
    empty slot (card.lua:3966-3972)."""

    def test_two_stencils_on_a_full_board_give_x2_each(self):
        gs = select_blind(new_run("AUDIT_STENCIL_FULL"))
        for key in ("j_stencil", "j_stencil", "j_greedy_joker", "j_lusty_joker", "j_mad"):
            give_joker(gs, key)
        play(gs, give_card(gs, "S_2"))
        # High Card: 5 + 2 chips; each Stencil: x_mult = (5 - 5) + 2 = 2
        assert gs["last_score_result"].total == (5 + 2) * 2 * 2

    def test_one_stencil_on_a_full_board_gives_nothing(self):
        gs = select_blind(new_run("AUDIT_STENCIL_ONE"))
        for key in ("j_stencil", "j_greedy_joker", "j_lusty_joker", "j_mad", "j_gluttenous_joker"):
            give_joker(gs, key)
        play(gs, give_card(gs, "S_2"))
        # x_mult = (5 - 5) + 1 = 1: not above 1, and no slot is empty
        assert gs["last_score_result"].total == 5 + 2


class TestSetAbilityStampsHandsPlayed:
    """G63: Card:set_ability sets ability.hands_played_at_create = G.GAME.hands_played on every
    call (card.lua:337). Midas Mask and Vampire call it in the before pass (card.lua:3448,
    3471), before the play is counted (state_events.lua:523)."""

    def test_midas_mask(self):
        gs = select_blind(new_run("AUDIT_MIDAS_STAMP"))
        give_joker(gs, "j_midas_mask")
        play(gs, give_card(gs, "S_2"))
        play(gs, give_card(gs, "H_3"))
        king = give_card(gs, "S_K")
        play(gs, king)
        assert king.center_key == "m_gold"
        assert king.ability["hands_played_at_create"] == 2

    def test_vampire(self):
        gs = select_blind(new_run("AUDIT_VAMPIRE_STAMP"))
        give_joker(gs, "j_vampire")
        play(gs, give_card(gs, "S_2"))
        card = give_card(gs, "H_3", enhancement="m_bonus")
        play(gs, card)
        assert card.center_key == "c_base"
        assert card.ability["hands_played_at_create"] == 1


class TestTheArmPreview:
    """G64: Blind:debuff_hand runs The Arm's test in check mode as well (blind.lua:550-559):
    highlighting a hand above level 1 sets blind.triggered, without the demote. A discard keeps
    the flag the highlight left (parse_highlighted, cardarea.lua:162-185)."""

    def test_discarding_a_levelled_hand_leaves_the_arm_triggered(self):
        gs = new_run("AUDIT_ARM_PREVIEW")
        force_boss(gs, "bl_arm")
        select_blind(gs)
        gs["hand_levels"].level_up("Pair")
        discard(gs, give_card(gs, "S_5"), give_card(gs, "H_5"))
        assert gs["blind"].triggered
        assert gs["hand_levels"]["Pair"].level == 2

    def test_discarding_a_level_one_hand_does_not(self):
        gs = new_run("AUDIT_ARM_PREVIEW_L1")
        force_boss(gs, "bl_arm")
        select_blind(gs)
        discard(gs, give_card(gs, "S_5"), give_card(gs, "H_5"))
        assert not gs["blind"].triggered

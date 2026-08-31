import torch

from model.policy_value_net import PolicyValueNet
from scripts.arena import _dedupe_move_lists, load_player


def _save_checkpoint(tmp_path):
    # load_player always constructs PolicyValueNet(board_size=8) with the default
    # num_blocks, so the saved state dict must match that shape.
    model = PolicyValueNet(board_size=8)
    model._init_weights()
    path = tmp_path / "ckpt.pth"
    torch.save({"model_state_dict": model.state_dict()}, path)
    return str(path)


def test_load_player_default_has_no_schedule_bonus(tmp_path):
    """Regression test for a real bug: arena used to hard-code use_schedule=True for the
    candidate and False for the baseline, giving the candidate a search-time c_puct bonus
    (up to 4.0 vs. a flat 1.5) regardless of which model was actually being tested - meaning
    every arena "win" this project ever recorded was confounded by this asymmetry, not just
    reflecting trained-model strength. Both sides must default to no schedule."""
    ckpt = _save_checkpoint(tmp_path)
    player = load_player(ckpt, "cpu", sims=1, use_schedule=False, schedule=None)
    assert player.mcts.c_puct_schedule == {"enabled": False}
    for depth in range(4):
        assert player.mcts._effective_c_puct(depth) == player.mcts.c_puct


def test_load_player_schedule_only_when_requested(tmp_path):
    ckpt = _save_checkpoint(tmp_path)
    schedule = {"enabled": True, "c0": 2.5, "lambda_": 0.3, "c_min": 1.0}
    scheduled = load_player(ckpt, "cpu", sims=1, use_schedule=True, schedule=schedule)
    unscheduled = load_player(ckpt, "cpu", sims=1, use_schedule=False, schedule=None)

    # With the schedule on, depth 0's effective c_puct is boosted above the base constant;
    # without it, effective c_puct is just the flat base constant at every depth.
    assert scheduled.mcts._effective_c_puct(0) > scheduled.mcts.c_puct
    assert unscheduled.mcts._effective_c_puct(0) == unscheduled.mcts.c_puct


def test_dedupe_move_lists_collapses_identical_trajectories():
    """Regression test for the trajectory-independence finding: games sharing a byte-identical
    move sequence (a real, observed consequence of stochastic eval only randomizing plies 0-1,
    with deterministic MCTS after) must collapse to one representative, not be double-counted
    as independent trials."""
    a = [(2, 3), (4, 5)]
    b = [(2, 3), (4, 5)]  # identical to a
    c = [(2, 3), (4, 6)]  # differs at the second move

    reps = _dedupe_move_lists([a, b, c])
    assert reps == [0, 2]


def test_dedupe_move_lists_all_unique():
    reps = _dedupe_move_lists([[(0, 0)], [(0, 1)], [(0, 2)]])
    assert reps == [0, 1, 2]


def test_dedupe_move_lists_empty():
    assert _dedupe_move_lists([]) == []

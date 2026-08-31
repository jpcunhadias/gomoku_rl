# scripts/xai_attributions.py
"""
Captum attribution comparison between two checkpoints (default: Cycle 1 vs Cycle 2), on real
board positions captured from actual games between them.

Why real positions: scripts/arena.py never logged move sequences, only aggregate win/loss/draw
stats, so no exact position from the already-completed C1-vs-C2 headline arena can be recovered.
This script captures fresh games under the same conditions (arena.load_player/play_one, with the
schedule-symmetry fix and the same stochastic-eval settings as the real headline arena) and picks
a small, labeled set of positions from them for attribution, rather than an arbitrary board.

For each selected position and each checkpoint, computes Integrated Gradients attribution
(Captum) for:
  - the policy head, w.r.t. the logit of that network's own top *legal* move
  - the value head, w.r.t. its scalar output

Usage:
  # Fresh capture (plays games, may take a few minutes depending on --games/--sims):
  PYTHONPATH=. uv run python scripts/xai_attributions.py \
      --c1_ckpt checkpoints/models/c1_cycle1_best.pth \
      --c2_ckpt checkpoints/models/c1_cycle2_best.pth

  # Reuse a previous capture, skip replaying games:
  PYTHONPATH=. uv run python scripts/xai_attributions.py \
      --positions_file checkpoints/xai/games.json
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from captum.attr import IntegratedGradients

from game.encoder import board_to_tensor
from game.gomoku import GomokuBoard
from model.policy_value_net import PolicyValueNet
from scripts.arena import load_player, play_one
from utils.seeding import set_global_seed

# Matches the real headline arena's stochastic-eval config (scripts/arena.py defaults).
EVAL_ROOT_EPS = 0.12
EVAL_TAU0 = 0.08
EVAL_TAU1 = 0.05


def capture_games(c1_ckpt, c2_ckpt, device, games, sims, seed):
    """Play `games` real games between the two checkpoints under fair, symmetric-schedule,
    stochastic-eval conditions (matching the reconfirmed headline arena), recording every
    move. Returns a list of {"moves": [...], "black": "c1"|"c2", "white": "c1"|"c2",
    "winner": 0|1|2}.
    """
    set_global_seed(seed)
    pairs = max(1, games // 2)
    results = []

    for i in range(pairs):
        p_c1 = load_player(
            c1_ckpt, device, sims, use_schedule=False, stochastic=True,
            root_eps=EVAL_ROOT_EPS, tau0=EVAL_TAU0, tau1=EVAL_TAU1, name="c1",
        )
        p_c2 = load_player(
            c2_ckpt, device, sims, use_schedule=False, stochastic=True,
            root_eps=EVAL_ROOT_EPS, tau0=EVAL_TAU0, tau1=EVAL_TAU1, name="c2",
        )
        hist_a: list = []
        w_a = int(play_one(p_c2, p_c1, stochastic=True, history=hist_a))
        results.append({"moves": hist_a, "black": "c2", "white": "c1", "winner": w_a})

        p_c1 = load_player(
            c1_ckpt, device, sims, use_schedule=False, stochastic=True,
            root_eps=EVAL_ROOT_EPS, tau0=EVAL_TAU0, tau1=EVAL_TAU1, name="c1",
        )
        p_c2 = load_player(
            c2_ckpt, device, sims, use_schedule=False, stochastic=True,
            root_eps=EVAL_ROOT_EPS, tau0=EVAL_TAU0, tau1=EVAL_TAU1, name="c2",
        )
        hist_b: list = []
        w_b = int(play_one(p_c1, p_c2, stochastic=True, history=hist_b))
        results.append({"moves": hist_b, "black": "c1", "white": "c2", "winner": w_b})

        print(f"[capture] pair {i + 1}/{pairs}: game A winner={w_a}, game B winner={w_b}")

    return results


def reconstruct_board(moves):
    """Replay a move list from an empty board."""
    board = GomokuBoard()
    for r, c in moves:
        board.apply_move(r, c)
    return board


def select_positions(games):
    """Pick a small, labeled set of positions from captured games, not an exhaustive sweep."""
    positions = [{"label": "ply0_empty", "moves": [], "note": "shared starting position"}]

    decisive = [g for g in games if g["winner"] != 0]
    draws = [g for g in games if g["winner"] == 0]

    if decisive:
        g = decisive[0]
        moves = g["moves"]
        positions.append({
            "label": "ply1", "moves": moves[:1],
            "note": f"after {g['black']}'s (Black) opening move",
        })
        mid = len(moves) // 2
        positions.append({
            "label": "midgame", "moves": moves[:mid],
            "note": f"ply {mid} of a decisive game (black={g['black']}, winner={g['winner']})",
        })
        positions.append({
            "label": "pre_winning_move", "moves": moves[:-1],
            "note": f"one ply before the winning move (black={g['black']}, winner={g['winner']})",
        })

    if draws:
        g = draws[0]
        moves = g["moves"]
        mid = len(moves) // 2
        positions.append({
            "label": "draw_midgame", "moves": moves[:mid],
            "note": f"ply {mid} of a drawn game (black={g['black']})",
        })

    return positions


def top_legal_move_index(policy_logits, board):
    """Index (row-major, matching board.move_to_index) of the highest-logit *legal* move."""
    legal_idx = board.get_legal_move_indices()
    logits = policy_logits.detach().cpu().numpy().reshape(-1)
    best = max(legal_idx, key=lambda i: logits[i])
    return best


def attribute_position(model, board):
    """Run Integrated Gradients on the policy (top legal move) and value heads for one
    position under one model. Returns a dict of predictions + attribution arrays."""
    device = next(model.parameters()).device
    x = board_to_tensor(board, board.current_player).unsqueeze(0).to(device)
    x.requires_grad_(True)

    with torch.no_grad():
        policy_logits, value = model(x)
    top_idx = top_legal_move_index(policy_logits, board)
    top_move = board.index_to_move(top_idx)
    legal_idx = board.get_legal_move_indices()
    probs = torch.softmax(policy_logits.view(-1), dim=0).detach().cpu().numpy()
    top_prob = float(probs[top_idx] / sum(probs[i] for i in legal_idx))

    # Baseline: no stones on either plane; the constant turn-indicator plane (channel 2) is left
    # matching the real input, since an all-zero turn plane is a state the network never sees.
    # (A side effect: IG's attribution for channel 2 is analytically ~0, since baseline == input
    # there — not an empirical finding, just confirms IG can't attribute an unchanging channel.)
    baseline = x.detach().clone()
    baseline[:, 0, :, :] = 0.0
    baseline[:, 1, :, :] = 0.0

    def policy_forward(inp):
        return model(inp)[0]

    def value_forward(inp):
        return model(inp)[1]

    ig_policy = IntegratedGradients(policy_forward)
    policy_attr = ig_policy.attribute(x, baselines=baseline, target=top_idx)
    policy_attr = policy_attr.detach().cpu().numpy()[0]  # [3, 8, 8]

    ig_value = IntegratedGradients(value_forward)
    value_attr = ig_value.attribute(x, baselines=baseline)
    value_attr = value_attr.detach().cpu().numpy()[0]  # [3, 8, 8]

    return {
        "top_move": list(top_move),
        "top_move_prob": top_prob,
        "value": float(value.item()),
        "policy_attr": policy_attr.tolist(),
        "value_attr": value_attr.tolist(),
        "channel2_attr_abs_max": {
            "policy": float(np.abs(policy_attr[2]).max()),
            "value": float(np.abs(value_attr[2]).max()),
        },
    }


def plot_comparison(position, result_c1, result_c2, board, out_path):
    """One figure per position: rows = (own-stone attr, opponent-stone attr), columns =
    (Cycle 1, Cycle 2), for a single head (policy or value already selected by caller via
    `result_c1["policy_attr"]` or `result_c1["value_attr"]`)."""
    attr_c1 = np.array(result_c1)  # [3, 8, 8]
    attr_c2 = np.array(result_c2)

    vmax = max(np.abs(attr_c1[:2]).max(), np.abs(attr_c2[:2]).max(), 1e-8)

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    channel_names = ["own-stone plane", "opponent-stone plane"]
    for row, ch_name in enumerate(channel_names):
        for col, (name, attr) in enumerate([("Cycle 1", attr_c1), ("Cycle 2", attr_c2)]):
            ax = axes[row, col]
            im = ax.imshow(attr[row], cmap="coolwarm", vmin=-vmax, vmax=vmax)
            for r in range(board.board_size):
                for c in range(board.board_size):
                    stone = board.board[r, c]
                    if stone == 1:
                        ax.text(c, r, "X", ha="center", va="center", fontsize=10)
                    elif stone == 2:
                        ax.text(c, r, "O", ha="center", va="center", fontsize=10)
            ax.set_title(f"{name} — {ch_name}")
            ax.set_xticks([])
            ax.set_yticks([])
    fig.colorbar(im, ax=axes, shrink=0.7)
    fig.suptitle(position["label"])
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--c1_ckpt", type=str, default="checkpoints/models/c1_cycle1_best.pth")
    ap.add_argument("--c2_ckpt", type=str, default="checkpoints/models/c1_cycle2_best.pth")
    ap.add_argument("--games", type=int, default=10, help="Games to capture (ignored if --positions_file given)")
    ap.add_argument("--sims", type=int, default=400, help="MCTS sims/move during capture")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--positions_file", type=str, default=None, help="Reuse a previous games.json instead of capturing fresh games")
    ap.add_argument("--games_out", type=str, default="checkpoints/xai/games.json")
    ap.add_argument("--out_json", type=str, default="checkpoints/xai/attributions.json")
    ap.add_argument("--plots_dir", type=str, default="checkpoints/xai/plots")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(args.games_out), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)

    if args.positions_file:
        print(f"[xai] loading captured games from {args.positions_file}")
        with open(args.positions_file) as f:
            games = json.load(f)
    else:
        print(f"[xai] capturing {args.games} fresh games (sims={args.sims}) between {args.c1_ckpt} and {args.c2_ckpt}")
        games = capture_games(args.c1_ckpt, args.c2_ckpt, device, args.games, args.sims, args.seed)
        with open(args.games_out, "w") as f:
            json.dump(games, f, indent=2)
        print(f"[xai] wrote {args.games_out}")

    positions = select_positions(games)
    print(f"[xai] selected {len(positions)} positions: {[p['label'] for p in positions]}")

    model_c1 = PolicyValueNet.load_from_checkpoint(args.c1_ckpt, device=device)
    model_c2 = PolicyValueNet.load_from_checkpoint(args.c2_ckpt, device=device)

    report = []
    for pos in positions:
        board = reconstruct_board(pos["moves"])
        res_c1 = attribute_position(model_c1, board)
        res_c2 = attribute_position(model_c2, board)

        plot_comparison(pos, res_c1["policy_attr"], res_c2["policy_attr"], board,
                         os.path.join(args.plots_dir, f"{pos['label']}_policy.png"))
        plot_comparison(pos, res_c1["value_attr"], res_c2["value_attr"], board,
                         os.path.join(args.plots_dir, f"{pos['label']}_value.png"))

        report.append({
            "label": pos["label"],
            "note": pos["note"],
            "moves": pos["moves"],
            "c1": res_c1,
            "c2": res_c2,
        })

        print(f"[xai] {pos['label']}: c1 top={res_c1['top_move']} (p={res_c1['top_move_prob']:.3f}, v={res_c1['value']:.3f})"
              f" | c2 top={res_c2['top_move']} (p={res_c2['top_move_prob']:.3f}, v={res_c2['value']:.3f})")

    with open(args.out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[xai] wrote {args.out_json}")
    print(f"[xai] plots in {args.plots_dir}")


if __name__ == "__main__":
    main()

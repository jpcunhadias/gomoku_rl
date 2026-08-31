# scripts/xai_attributions.py
"""
Attribution comparison between two checkpoints (default: Cycle 1 vs Cycle 2), on real board
positions captured from actual games between them, cross-checked across two independent
attribution methods (Captum Integrated Gradients and SHAP GradientExplainer).

Why real positions: scripts/arena.py never logged move sequences, only aggregate win/loss/draw
stats, so no exact position from the already-completed C1-vs-C2 headline arena can be recovered.
This script captures fresh games under the same conditions (arena.load_player/play_one, with the
schedule-symmetry fix and the same stochastic-eval settings as the real headline arena) and picks
several labeled positions per category (opening, midgame, pre-winning-move, draw) from multiple
games, rather than a single example of each.

Why two methods: this project's own history (the arena schedule confound) is a standing argument
against trusting a single tool's output uncross-checked. SHAP's GradientExplainer uses a
background sample of other real positions (not a single zero baseline) as its reference
distribution, so agreement between it and Integrated Gradients on the same position is a real,
independent check, not just running the same math twice.

For each selected position and each checkpoint, computes attribution for:
  - the policy head, w.r.t. the logit of that network's own top *legal* move
  - the value head, w.r.t. its scalar output

Usage:
  # Fresh capture (plays games, can take a while depending on --games/--sims):
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
import random

import matplotlib.pyplot as plt
import numpy as np
import shap
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


def select_positions(games, n_decisive=5, n_draws=3):
    """Pick a labeled set of positions from captured games. Draws several examples per category
    (from different games) rather than one, so later analysis isn't resting on n=1 per category."""
    positions = [{"label": "ply0_empty", "moves": [], "note": "shared starting position"}]

    decisive = [g for g in games if g["winner"] != 0]
    draws = [g for g in games if g["winner"] == 0]

    for i, g in enumerate(decisive[:n_decisive]):
        moves = g["moves"]
        positions.append({
            "label": f"decisive{i}_ply1", "moves": moves[:1],
            "note": f"after {g['black']}'s (Black) opening move",
        })
        mid = len(moves) // 2
        positions.append({
            "label": f"decisive{i}_midgame", "moves": moves[:mid],
            "note": f"ply {mid} of a decisive game (black={g['black']}, winner={g['winner']})",
        })
        positions.append({
            "label": f"decisive{i}_pre_winning_move", "moves": moves[:-1],
            "note": f"one ply before the winning move (black={g['black']}, winner={g['winner']})",
        })

    for i, g in enumerate(draws[:n_draws]):
        moves = g["moves"]
        mid = len(moves) // 2
        positions.append({
            "label": f"draw{i}_midgame", "moves": moves[:mid],
            "note": f"ply {mid} of a drawn game (black={g['black']})",
        })

    return positions


def build_shap_background(games, size, seed):
    """Sample a pool of real board states (various plies, various games) to use as SHAP
    GradientExplainer's reference distribution, instead of a single zero baseline."""
    rng = random.Random(seed)
    candidates = []
    for g in games:
        moves = g["moves"]
        if len(moves) < 2:
            continue
        # A handful of plies per game, skipping ply 0 (that's the degenerate empty board).
        for ply in range(1, len(moves), max(1, len(moves) // 4)):
            candidates.append(moves[:ply])
    rng.shuffle(candidates)
    chosen = candidates[:size]
    tensors = []
    for moves in chosen:
        board = reconstruct_board(moves)
        tensors.append(board_to_tensor(board, board.current_player))
    return torch.stack(tensors) if tensors else torch.zeros(1, 3, 8, 8)


def top_legal_move_index(policy_logits, board):
    """Index (row-major, matching board.move_to_index) of the highest-logit *legal* move."""
    legal_idx = board.get_legal_move_indices()
    logits = policy_logits.detach().cpu().numpy().reshape(-1)
    best = max(legal_idx, key=lambda i: logits[i])
    return best


class _SingleOutput(torch.nn.Module):
    """Wraps a closure returning one tensor, so SHAP's GradientExplainer (which expects an
    nn.Module) can be pointed at an arbitrary slice of PolicyValueNet's forward output."""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


def _cosine_similarity(a, b):
    a, b = a.reshape(-1), b.reshape(-1)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def attribute_position(model, board, shap_background, shap_nsamples=50):
    """Run Integrated Gradients AND SHAP (GradientExplainer) on the policy (top legal move) and
    value heads for one position under one model, as a cross-check between two independent
    attribution methods. Returns a dict of predictions + both methods' attribution arrays."""
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

    # IG baseline: no stones on either plane; the constant turn-indicator plane (channel 2) is
    # left matching the real input, since an all-zero turn plane is a state the network never
    # sees. (Side effect: IG's attribution for channel 2 is analytically ~0, since baseline ==
    # input there — not an empirical finding, just confirms IG can't attribute an unchanging
    # channel.)
    baseline = x.detach().clone()
    baseline[:, 0, :, :] = 0.0
    baseline[:, 1, :, :] = 0.0

    def policy_forward(inp):
        return model(inp)[0]

    def value_forward(inp):
        return model(inp)[1]

    ig_policy = IntegratedGradients(policy_forward)
    ig_policy_attr = ig_policy.attribute(x, baselines=baseline, target=top_idx)
    ig_policy_attr = ig_policy_attr.detach().cpu().numpy()[0]  # [3, 8, 8]

    ig_value = IntegratedGradients(value_forward)
    ig_value_attr = ig_value.attribute(x, baselines=baseline)
    ig_value_attr = ig_value_attr.detach().cpu().numpy()[0]  # [3, 8, 8]

    # SHAP: same fixed target (this network's own top legal move) for apples-to-apples
    # comparison against IG, but a real background *sample* of other positions as the reference
    # distribution rather than a single zero baseline.
    background = shap_background.to(device)
    policy_wrap = _SingleOutput(lambda inp: model(inp)[0][:, top_idx : top_idx + 1])
    shap_policy_expl = shap.GradientExplainer(policy_wrap, background)
    shap_policy_attr = shap_policy_expl.shap_values(x, nsamples=shap_nsamples)[0, ..., 0]

    value_wrap = _SingleOutput(lambda inp: model(inp)[1])
    shap_value_expl = shap.GradientExplainer(value_wrap, background)
    shap_value_attr = shap_value_expl.shap_values(x, nsamples=shap_nsamples)[0, ..., 0]

    return {
        "top_move": list(top_move),
        "top_move_prob": top_prob,
        "value": float(value.item()),
        "ig_policy_attr": ig_policy_attr.tolist(),
        "ig_value_attr": ig_value_attr.tolist(),
        "shap_policy_attr": shap_policy_attr.tolist(),
        "shap_value_attr": shap_value_attr.tolist(),
        "channel2_attr_abs_max": {
            "ig_policy": float(np.abs(ig_policy_attr[2]).max()),
            "ig_value": float(np.abs(ig_value_attr[2]).max()),
        },
        "ig_vs_shap_cosine": {
            # Own-stone/opponent-stone channels only (0:2) — channel 2 is constant, excluding it
            # avoids a shared-zero coordinate inflating apparent agreement.
            "policy": _cosine_similarity(ig_policy_attr[:2], shap_policy_attr[:2]),
            "value": _cosine_similarity(ig_value_attr[:2], shap_value_attr[:2]),
        },
    }


def plot_comparison(position, method_label, result_c1, result_c2, board, out_path):
    """One figure per (position, method, head): rows = (own-stone attr, opponent-stone attr),
    columns = (Cycle 1, Cycle 2)."""
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
    fig.suptitle(f"{position['label']} ({method_label})")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--c1_ckpt", type=str, default="checkpoints/models/c1_cycle1_best.pth")
    ap.add_argument("--c2_ckpt", type=str, default="checkpoints/models/c1_cycle2_best.pth")
    ap.add_argument("--games", type=int, default=40, help="Games to capture (ignored if --positions_file given)")
    ap.add_argument("--sims", type=int, default=400, help="MCTS sims/move during capture")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_decisive_examples", type=int, default=5, help="Decisive games to draw ply1/midgame/pre_winning_move positions from")
    ap.add_argument("--n_draw_examples", type=int, default=3, help="Drawn games to draw a midgame position from")
    ap.add_argument("--shap_background_size", type=int, default=20, help="Real positions sampled as SHAP's reference distribution")
    ap.add_argument("--shap_nsamples", type=int, default=50, help="SHAP GradientExplainer interpolation samples per call")
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

    positions = select_positions(games, n_decisive=args.n_decisive_examples, n_draws=args.n_draw_examples)
    print(f"[xai] selected {len(positions)} positions: {[p['label'] for p in positions]}")

    shap_background = build_shap_background(games, args.shap_background_size, args.seed)
    print(f"[xai] SHAP background: {shap_background.shape[0]} real positions")

    model_c1 = PolicyValueNet.load_from_checkpoint(args.c1_ckpt, device=device)
    model_c2 = PolicyValueNet.load_from_checkpoint(args.c2_ckpt, device=device)

    report = []
    for pos in positions:
        board = reconstruct_board(pos["moves"])
        res_c1 = attribute_position(model_c1, board, shap_background, args.shap_nsamples)
        res_c2 = attribute_position(model_c2, board, shap_background, args.shap_nsamples)

        for method in ("ig", "shap"):
            for head in ("policy", "value"):
                plot_comparison(
                    pos, method,
                    res_c1[f"{method}_{head}_attr"], res_c2[f"{method}_{head}_attr"], board,
                    os.path.join(args.plots_dir, f"{pos['label']}_{method}_{head}.png"),
                )

        report.append({
            "label": pos["label"],
            "note": pos["note"],
            "moves": pos["moves"],
            "c1": res_c1,
            "c2": res_c2,
        })

        print(f"[xai] {pos['label']}: c1 top={res_c1['top_move']} (p={res_c1['top_move_prob']:.3f}, v={res_c1['value']:.3f})"
              f" | c2 top={res_c2['top_move']} (p={res_c2['top_move_prob']:.3f}, v={res_c2['value']:.3f})"
              f" | IG-vs-SHAP cosine: c1 policy={res_c1['ig_vs_shap_cosine']['policy']:.2f} value={res_c1['ig_vs_shap_cosine']['value']:.2f}"
              f", c2 policy={res_c2['ig_vs_shap_cosine']['policy']:.2f} value={res_c2['ig_vs_shap_cosine']['value']:.2f}")

    with open(args.out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[xai] wrote {args.out_json}")
    print(f"[xai] plots in {args.plots_dir}")


if __name__ == "__main__":
    main()

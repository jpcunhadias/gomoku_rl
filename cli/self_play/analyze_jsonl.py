import json
import os
from collections import defaultdict

import numpy as np

from train.config import get_config

PATH = "checkpoints/selfplay/selfplay_v2.jsonl"
# Dynamically fetch tau_cutoff_plies from the configuration
TAU_CUTOFF_PLIES = get_config().tau_cutoff_plies


def load_jsonl(path):
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # skip malformed lines silently
                continue


def main():
    if not os.path.exists(PATH):
        raise FileNotFoundError(f"JSONL not found: {PATH}")

    summaries = []
    # Optional budget sanity (per-move records)
    sims_by_phase = defaultdict(list)

    # We'll recompute early-game medians from move records to get normalized H,
    # then compare with any game_summary values you already have.
    early_medians_raw = []
    early_medians_norm = []

    # Buffers for one game's early entropies
    cur_game_early_raw = []
    cur_game_early_norm = []
    have_any_move = False

    def flush_game():
        if cur_game_early_raw:
            early_medians_raw.append(float(np.median(cur_game_early_raw)))
            early_medians_norm.append(float(np.median(cur_game_early_norm)))

    for rec in load_jsonl(PATH):
        # Per-move record?
        if "move_number" in rec and "entropy_pi_mcts" in rec:
            have_any_move = True
            mv = int(rec["move_number"])
            # derive n_legal from legal_mask if present, else skip normalized
            n_legal = None
            if "legal_mask" in rec:
                # legal_mask was saved as a nested list of booleans; count True
                lm = rec["legal_mask"]
                # robust count even if it's a flat list or nested
                if isinstance(lm, list) and lm and isinstance(lm[0], list):
                    n_legal = sum(sum(1 for x in row if x) for row in lm)
                elif isinstance(lm, list):
                    n_legal = sum(1 for x in lm if x)

            H_raw = float(rec["entropy_pi_mcts"])
            if mv < TAU_CUTOFF_PLIES:
                cur_game_early_raw.append(H_raw)
                if n_legal and n_legal > 1:
                    cur_game_early_norm.append(H_raw / np.log(n_legal))

            # sim budget snapshot by rough phase bands (keep your cutoffs)
            if mv < 12:
                sims_by_phase["early"].append(rec.get("sims"))
            elif mv < 28:
                sims_by_phase["mid"].append(rec.get("sims"))
            else:
                sims_by_phase["late"].append(rec.get("sims"))

        # Game summary marks end of a game → flush accumulators
        if rec.get("type") == "game_summary":
            summaries.append(rec)
            flush_game()
            cur_game_early_raw = []
            cur_game_early_norm = []
            have_any_move = False

    # If file ended mid-game without a summary
    if have_any_move:
        flush_game()

    total = len(early_medians_raw)

    def iqr(xs):
        if not xs:
            return (float("nan"), float("nan"), float("nan"))
        q1, med, q3 = np.percentile(xs, [25, 50, 75])
        return (float(med), float(q1), float(q3))

    med_raw, q1_raw, q3_raw = iqr(early_medians_raw)
    med_norm, q1_norm, q3_norm = iqr(early_medians_norm)

    # Unique openings via open5_key (from summaries if present)
    keys = [s.get("open5_key") for s in summaries if s.get("open5_key") is not None]
    uniq = len(set(keys))
    total_games = len(summaries) if summaries else total
    uniq_rate = (uniq / total_games * 100.0) if total_games else 0.0

    print("\n=== Phase B summary ===")
    print(f"Games analyzed: {total_games or total}")
    print(
        f"Early-entropy (RAW) median-of-medians: {med_raw:.3f} (IQR {q1_raw:.3f}–{q3_raw:.3f})"
    )
    print(
        f"Early-entropy (NORMALIZED) median-of-medians: {med_norm:.3f} (IQR {q1_norm:.3f}–{q3_norm:.3f})"
    )
    if total_games:
        print(f"Unique opening-5 keys: {uniq} / {total_games}  ({uniq_rate:.1f}%)")

    # ---- Budget sanity (optional)
    def stat(xs):
        xs = [x for x in xs if x is not None]
        if not xs:
            return "n=0"
        xs = np.asarray(xs)
        return f"n={len(xs)} | med={np.median(xs):.1f} | p25={np.percentile(xs, 25):.1f} | p75={np.percentile(xs, 75):.1f}"

    print("\nSimulation budget by phase (from per-move recs):")
    for ph in ["early", "mid", "late"]:
        print(f"  {ph:5s}: {stat(sims_by_phase[ph])}")

    # ---- Guidance based on *normalized* band
    print("\n=== Guidance (uses NORMALIZED early entropy) ===")
    if np.isnan(med_norm):
        print(
            "No normalized entropy values found. Ensure per-move logs include legal_mask and entropy_pi_mcts."
        )
    elif med_norm < 0.45:
        print("Normalized entropy low (<0.45): increase exploration slightly.")
        print("  • Try dirichlet_epsilon: +0.05")
        print("  • Or raise dirichlet_alpha_max a bit")
        print("  • Optionally increase tau_cutoff_plies by 1–2")
    elif med_norm > 0.65:
        print("Normalized entropy high (>0.65): decrease exploration slightly.")
        print("  • Try dirichlet_epsilon: -0.05")
        print("  • Or lower dirichlet_alpha_max a bit")
        print("  • Optionally reduce tau_cutoff_plies by 1–2")
    else:
        print("Normalized entropy in target band [0.45, 0.65]. ✅")

    print("\nNext: rerun policy_head_check.py to ensure Top-3 vs MCTS didn’t regress.")
    print(
        "      If unique openings < +25% vs baseline, nudge exploration up.", flush=True
    )


if __name__ == "__main__":
    main()

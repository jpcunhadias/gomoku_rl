#!/usr/bin/env python3
"""
Analyze exploration metrics from selfplay data.

Usage:
    python scripts/analyze_exploration.py checkpoints/selfplay/selfplay_c1_cycle2.jsonl
"""

import argparse
import json
import statistics
from collections import defaultdict


def analyze_selfplay(jsonl_path):
    """Analyze exploration metrics from selfplay JSONL file."""
    
    # Load data
    with open(jsonl_path, 'r') as f:
        lines = f.readlines()
    
    game_summaries = []
    training_samples = []
    
    for line in lines:
        data = json.loads(line)
        if data.get('type') == 'game_summary':
            game_summaries.append(data)
        else:
            training_samples.append(data)
    
    print("="*80)
    print(f"EXPLORATION ANALYSIS: {jsonl_path}")
    print("="*80)
    print(f"\nGames: {len(game_summaries)}")
    print(f"Training samples: {len(training_samples)}")
    
    # Analyze by ply
    ply_stats = defaultdict(list)
    
    for sample in training_samples:
        ply = sample.get('move_number', -1)
        ply_stats[ply].append({
            'entropy_mcts': sample.get('entropy_pi_mcts', 0),
            'entropy_net': sample.get('entropy_pi_net', 0),
            'kl': sample.get('kl_net_mcts', 0),
            'tau': sample.get('tau', 0),
            'dirichlet_eps': sample.get('dirichlet_eps', 0),
            'dirichlet_alpha': sample.get('dirichlet_alpha', 0),
        })
    
    # Ply 0 detailed analysis
    print("\n" + "="*80)
    print("PLY 0 (FIRST MOVE) ANALYSIS")
    print("="*80)
    
    if 0 in ply_stats:
        stats = ply_stats[0]
        entropy_mcts = [s['entropy_mcts'] for s in stats]
        tau = [s['tau'] for s in stats]
        eps = [s['dirichlet_eps'] for s in stats]
        alpha = [s['dirichlet_alpha'] for s in stats]
        
        print(f"\nSamples: {len(stats)}")
        print(f"\nEntropy MCTS:")
        print(f"  Mean:   {statistics.mean(entropy_mcts):.4f}")
        print(f"  Median: {statistics.median(entropy_mcts):.4f}")
        print(f"  Stdev:  {statistics.stdev(entropy_mcts) if len(entropy_mcts) > 1 else 0:.4f}")
        print(f"  Min:    {min(entropy_mcts):.4f}")
        print(f"  Max:    {max(entropy_mcts):.4f}")
        
        if len(entropy_mcts) >= 4:
            q = statistics.quantiles(entropy_mcts, n=4)
            print(f"  Q1:     {q[0]:.4f}")
            print(f"  Q3:     {q[2]:.4f}")
        
        print(f"\nConfiguration:")
        print(f"  Tau:              {tau[0]:.4f}")
        print(f"  Dirichlet eps:    {eps[0]:.4f}")
        print(f"  Dirichlet alpha:  {alpha[0]:.4f}")
        
        # Distribution
        bins = [(0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 5.0)]
        print(f"\nEntropy Distribution:")
        for low, high in bins:
            count = sum(1 for e in entropy_mcts if low <= e < high)
            pct = 100 * count / len(entropy_mcts)
            bar = "█" * int(pct / 2)
            print(f"  {low:.1f}-{high:.1f}: {count:3d} ({pct:5.1f}%) {bar}")
        
        # Targets
        very_low = sum(1 for e in entropy_mcts if e < 0.5)
        low = sum(1 for e in entropy_mcts if 0.5 <= e < 1.0)
        problematic = very_low + low
        
        print(f"\n⚠️  Problematic samples (entropy < 1.0): {problematic}/{len(entropy_mcts)} ({100*problematic/len(entropy_mcts):.1f}%)")
        
        median_ent = statistics.median(entropy_mcts)
        mean_ent = statistics.mean(entropy_mcts)
        
        status_median = "✓" if median_ent >= 2.0 else "✗"
        status_mean = "✓" if mean_ent >= 2.2 else "✗"
        status_problematic = "✓" if problematic < len(entropy_mcts) * 0.05 else "✗"
        
        print(f"\nTargets:")
        print(f"  {status_median} Median entropy > 2.0: {median_ent:.2f}")
        print(f"  {status_mean} Mean entropy > 2.2: {mean_ent:.2f}")
        print(f"  {status_problematic} < 5% problematic: {100*problematic/len(entropy_mcts):.1f}%")
    
    # Early plies summary
    print("\n" + "="*80)
    print("EARLY PLIES (0-2) SUMMARY")
    print("="*80)
    
    for ply in range(3):
        if ply in ply_stats:
            stats = ply_stats[ply]
            ent = [s['entropy_mcts'] for s in stats]
            tau_val = stats[0]['tau']
            print(f"\nPly {ply}: mean={statistics.mean(ent):.3f}, median={statistics.median(ent):.3f}, tau={tau_val:.3f}")
    
    # Game-level stats
    print("\n" + "="*80)
    print("GAME-LEVEL EARLY ENTROPY")
    print("="*80)
    
    if game_summaries:
        early_ent = [g['early_entropy_mcts_median'] for g in game_summaries]
        print(f"\nEarly entropy MCTS median across {len(game_summaries)} games:")
        print(f"  Mean:   {statistics.mean(early_ent):.4f}")
        print(f"  Median: {statistics.median(early_ent):.4f}")
        print(f"  Stdev:  {statistics.stdev(early_ent):.4f}")
        print(f"  Min:    {min(early_ent):.4f}")
        print(f"  Max:    {max(early_ent):.4f}")
        
        low_games = sum(1 for e in early_ent if e < 1.5)
        status_games = "✓" if low_games < len(game_summaries) * 0.2 else "✗"
        
        print(f"\n{status_games} Games with early entropy < 1.5: {low_games}/{len(game_summaries)} ({100*low_games/len(game_summaries):.1f}%)")
        print(f"   Target: <20%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze exploration metrics from selfplay data")
    parser.add_argument("jsonl_path", help="Path to selfplay JSONL file")
    
    args = parser.parse_args()
    analyze_selfplay(args.jsonl_path)

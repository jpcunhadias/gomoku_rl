Running debug checks for CYCLE=2
python debug/value_head_check.py \
  --checkpoint checkpoints/models/c1_cycle2_last.pth \
  --buffer checkpoints/buffers/replay_c1_cycle2.pkl
[INFO] device=cuda, seed=42
[INFO] Loaded model: checkpoints/models/c1_cycle2_last.pth
[INFO] Loaded buffer with 23719 samples

=== VALUE HEAD CHECK ===
Pred shape: torch.Size([256, 1])  |  v̂ range: [-1.0000, 0.9999]  |  in [-1,1]? True
Targets present (counts): {-1.0: 102, 0.0: 34, 1.0: 120}

Brier score: 0.391245
Scalar ECE (10 bins): 0.225809

Calibration bins (bin_idx, count, mean_pred, mean_true, abs_gap):
   0    96   -0.9549   -0.8229    0.1320
   1     8   -0.6941   -0.1250    0.5691
   2    11   -0.5152    0.0909    0.6061
   3    10   -0.2985    0.3000    0.5985
   4    20   -0.0805    0.2500    0.3305
   5     5    0.0809    0.4000    0.3191
   6    10    0.2643    0.4000    0.1357
   7    10    0.5113    0.9000    0.3887
   8    20    0.6940    1.0000    0.3060
   9    66    0.9449    0.8182    0.1267

Threshold sanity:
  P(v̂>0.7 | z=+1): 0.558
  P(v̂<-0.7 | z=-1): 0.784
  P(|v̂|<0.2 | z=0): 0.176

Pre-tanh saturation: share(|pre_tanh|>2.0) = 0.359

=== CHECKBOX SUMMARY ===
[x] Value outputs in [-1,1] (approx)
[ ] Pre-tanh saturation < 20% (if measurable)
[x] Brier computed
[x] ECE computed
[x] Hist & scatter plots saved to: debug/debug_outputs
python debug/policy_head_check.py \
  --checkpoint checkpoints/models/c1_cycle2_last.pth \
  --buffer checkpoints/buffers/replay_c1_cycle2.pkl

=== POLICY HEAD CHECK ===
Batch size: 512  |  used (non-terminal): 512  |  skipped (terminal): 0
Legality violations (π>0 on illegal): 0
Normalization violations (sum(legal) != 1): 0
Normalized entropy H(π)/log(#legal): mean=0.929  median=0.970  IQR=[0.960,0.985]
KL(π_net || π_mcts) over legal:       mean=0.012  median=0.000  IQR=[0.000,0.000]
Top-1 vs MCTS argmax: 46.88%
Top-3 vs MCTS argmax: 72.46%

=== CHECKBOX SUMMARY ===
[x] π[illegal] == 0 for all used samples
[x] sum(π on legal) == 1 (±1e-6)
[ ] normalized entropy median in [0.45, 0.65] (exploration sanity)
[x] KL computed (baseline)
[x] Top-k computed (diagnostic)
python debug/training_smoke_check.py \
  --checkpoint checkpoints/models/c1_cycle2_last.pth \
  --buffer checkpoints/buffers/replay_c1_cycle2.pkl
[Smoke] device=cuda
[Smoke] ckpt=checkpoints/models/c1_cycle2_last.pth
[Smoke] buffer=checkpoints/buffers/replay_c1_cycle2.pkl
step   20 | policy 1.7490 | value 0.1443 | total 1.8212 | grad 3.983
step   40 | policy 1.7172 | value 0.1770 | total 1.8057 | grad 4.149
step   60 | policy 1.7222 | value 0.1574 | total 1.8009 | grad 4.177
step   80 | policy 1.4729 | value 0.1415 | total 1.5437 | grad 4.463
step  100 | policy 1.4723 | value 0.1498 | total 1.5473 | grad 4.516
step  120 | policy 1.3023 | value 0.1247 | total 1.3646 | grad 4.084
step  140 | policy 1.3060 | value 0.1343 | total 1.3732 | grad 4.468
step  160 | policy 1.2606 | value 0.1139 | total 1.3175 | grad 4.572
step  180 | policy 1.1292 | value 0.1082 | total 1.1833 | grad 4.325
step  200 | policy 1.1866 | value 0.1348 | total 1.2540 | grad 4.509

=== TRAINING SMOKE SUMMARY ===
Policy loss (first→last): 1.8954 → 1.1866
Value  loss (first→last): 0.1371 → 0.1348
Grad-norm median/95p: 4.335 / 4.910

=== CHECKBOX SUMMARY ===
[x] steps completed without NaNs/Inf
[x] Losses logged; basic decreasing trend expected (not strict)
[x] Grad norms within reasonable range (no monotonic blow-up)

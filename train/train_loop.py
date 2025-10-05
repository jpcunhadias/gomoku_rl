import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from model.policy_value_net import PolicyValueNet
from train.bucketer import BucketKey
from train.replay_buffer import ReplayBuffer


def _write_debug_log(path: Optional[str], content: str) -> None:
    """Helper method to write debug logs safely."""
    if path:
        try:
            with open(path, "a") as f:
                f.write(content)
        except Exception as e:
            print(f"[Debug Log Error] Could not write to {path}: {e}")


class AlphaZeroTrainer:
    """Trainer implementing the AlphaZero learning loop."""

    def __init__(
        self,
        model: PolicyValueNet,
        optimizer: optim.Optimizer,
        replay_buffer: ReplayBuffer,
        config: Any,
        device: str = "cpu",
        best_value_loss: float = float("inf"),
        save_paths: Optional[Dict[str, Path]] = None,
    ) -> None:
        """
        AlphaZero training loop for Gomoku.

        Args:
            model (PolicyValueNet): Your dual-head model.
            replay_buffer (ReplayBuffer): Self-play experience buffer.
            config (Namespace): Should contain batch_size, learning_rate, epochs, steps_per_epoch, save_path.
            device (str): 'cuda' or 'cpu'.
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.replay_buffer = replay_buffer
        self.config = config
        self.device = device
        self.batch_size = config.batch_size
        self.epochs = config.epochs
        self.steps_per_epoch = config.steps_per_epoch
        self.save_path = config.save_path
        self.save_paths = save_paths if save_paths else {}
        self.sidecar_jsonl = str(
            self.save_paths.get("sp_log", "checkpoints/selfplay/selfplay_v2.jsonl")
        )

        self.policy_loss_fn = nn.KLDivLoss(reduction="batchmean")
        self.value_loss_fn = nn.SmoothL1Loss(beta=1.0)
        self.best_value_loss = best_value_loss
        self.best_epoch = 0

        self.reload_buffer_every = getattr(config, "reload_buffer_every", 500)
        self.eval_every = getattr(config, "eval_every", 5)
        self.target_win_rate = getattr(config, "target_win_rate", 0.8)
        self.eval_num_games = getattr(config, "eval_num_games", 20)
        self.eval_num_simulations = getattr(config, "eval_num_simulations", 100)

        print(
            f"[Trainer Initialized] Evaluation every {self.eval_every} epochs, "
            f"target win rate = {self.target_win_rate * 100:.1f}%, "
            f"{self.eval_num_games} games per eval, "
            f"{self.eval_num_simulations} simulations per move."
        )

        self.writer = SummaryWriter(log_dir=os.path.join("logs", "train"))

        if getattr(config, "use_stratified_sampler", False):
            from train.stratified_sampler import TARGET_MIX, StratifiedBatchSampler

            self.sampler = StratifiedBatchSampler(
                sidecar_jsonl=self.sidecar_jsonl,
                buffer_len_fn=lambda: len(self.replay_buffer),
                target_mix=TARGET_MIX,
                refresh_every=max(1, self.steps_per_epoch // 2),
            )
        else:
            self.sampler = None

    def compute_loss(
        self,
        policy_logits: torch.Tensor,
        target_policy: torch.Tensor,
        value_pred: torch.Tensor,
        target_value: torch.Tensor,
    ) -> Tuple[torch.Tensor, float, float]:
        """
        Combines policy and value losses.

        Args:
            policy_logits (Tensor): Raw logits from policy head.
            target_policy (Tensor): Target probabilities (π from MCTS).
            value_pred (Tensor): Scalar value prediction in [-1, 1].
            target_value (Tensor): Game result encoded as -1 (loss), 0 (draw), 1 (win).
        """
        target_policy = target_policy.to(torch.float32)
        target_value = target_value.view(-1).to(torch.float32)

        log_probs = F.log_softmax(policy_logits, dim=1)
        policy_loss = self.policy_loss_fn(log_probs, target_policy)
        value_loss = self.value_loss_fn(value_pred.view(-1), target_value)

        total_loss = policy_loss * 1.0 + value_loss * 0.5

        return total_loss, policy_loss.item(), value_loss.item()

    def train(self, debug: bool = False) -> Tuple[Optional[int], float]:
        """
        Main training loop.
        """
        self.model.train()

        loss_log_path = (
            os.path.join("checkpoints", "train_loss_summary.log") if debug else None
        )
        stats_log_path = (
            os.path.join("checkpoints", "value_pred_stats.log") if debug else None
        )
        grad_log_path = (
            os.path.join("checkpoints", "gradient_debug.log") if debug else None
        )
        value_debug_path = (
            os.path.join("checkpoints", "value_pred_debug.log") if debug else None
        )

        for epoch in range(1, self.epochs + 1):
            epoch_policy_loss = 0.0
            epoch_value_loss = 0.0
            value_preds_this_epoch = []

            if epoch % self.reload_buffer_every == 0 and epoch != 1:
                print(
                    f"[Trainer] Reloading replay buffer from disk at epoch {epoch}..."
                )
                self.replay_buffer = ReplayBuffer.load("checkpoints/replay_buffer.pkl")
                if self.sampler is not None:
                    self.sampler.refresh()

            if self.sampler is not None:
                self.sampler.begin_epoch()

            for step in range(self.steps_per_epoch):
                if self.sampler is None:
                    states, target_policies, target_values = self.replay_buffer.sample(
                        self.batch_size
                    )
                    if set(torch.unique(target_values).tolist()) <= {0.0, 1.0, 2.0}:
                        target_values = target_values - 1.0
                    target_values = target_values.to(torch.float32)

                    states = states.to(self.device)
                    target_policies = target_policies.to(self.device)
                    target_values = target_values.to(self.device)
                else:
                    idxs = self.sampler.sample_indices(self.batch_size)
                    # in-place gather from buffer
                    batch = [self.replay_buffer.buffer[i] for i in idxs]
                    states, target_policies, target_values = zip(*batch)

                    states = torch.stack(states).to(self.device)
                    target_policies = torch.stack(target_policies).to(self.device)
                    target_values = torch.tensor(
                        target_values, dtype=torch.float32, device=self.device
                    )
                    if set(torch.unique(target_values).tolist()) <= {0.0, 1.0, 2.0}:
                        target_values = target_values - 1.0

                target_policies = target_policies.to(torch.float32)
                action_size = self.model.policy_fc.out_features
                target_policies = target_policies.view(-1, action_size)

                self.optimizer.zero_grad()
                logits, value_pred = self.model(states)

                if debug:
                    with torch.no_grad():
                        value_mean = value_pred.mean().item()
                        value_std = value_pred.std().item()
                        value_preds_this_epoch.append((value_mean, value_std))

                    _write_debug_log(
                        value_debug_path,
                        f"\nEpoch {epoch}, Step {step}:\n"
                        + "".join(
                            f"  z: {target}, v: {float(pred):.4f}\n"
                            for pred, target in zip(
                                value_pred.detach().cpu().view(-1).tolist()[:8],
                                target_values.detach().cpu().tolist()[:8],
                            )
                        ),
                    )

                loss, p_loss, v_loss = self.compute_loss(
                    logits, target_policies, value_pred, target_values
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                if debug:
                    _write_debug_log(
                        grad_log_path,
                        f"\nEpoch {epoch}, Step {step}:\n"
                        + "".join(
                            f"[{('Value' if 'value' in name else 'Policy')}] {name}: {param.grad.norm().item():.6f}\n"
                            for name, param in self.model.named_parameters()
                            if param.grad is not None
                        ),
                    )

                self.optimizer.step()
                epoch_policy_loss += p_loss
                epoch_value_loss += v_loss

            avg_p_loss = epoch_policy_loss / self.steps_per_epoch
            avg_v_loss = epoch_value_loss / self.steps_per_epoch

            if debug:
                _write_debug_log(
                    loss_log_path,
                    f"Epoch {epoch}: Policy Loss = {avg_p_loss:.4f}, Value Loss = {avg_v_loss:.4f}\n",
                )

                mean_of_means = sum(x[0] for x in value_preds_this_epoch) / len(
                    value_preds_this_epoch
                )
                mean_of_stds = sum(x[1] for x in value_preds_this_epoch) / len(
                    value_preds_this_epoch
                )
                _write_debug_log(
                    stats_log_path,
                    f"Epoch {epoch}: value_pred mean = {mean_of_means:.4f}, std = {mean_of_stds:.4f}\n",
                )

            print(
                f"Epoch {epoch}: Policy Loss = {avg_p_loss:.4f}, Value Loss = {avg_v_loss:.4f}"
            )

            if self.sampler is not None:
                total_examples = self.batch_size * self.steps_per_epoch
                target_mix = self.sampler.target_mix  # normalized {BucketKey: frac}

                requested_counts = {
                    k: int(round(v * total_examples)) for k, v in target_mix.items()
                }
                realized_counts = self.sampler.end_epoch_report()

                for k in target_mix.keys():
                    requested_counts.setdefault(k, 0)
                    realized_counts.setdefault(k, 0)

                req_frac = {
                    k: requested_counts[k] / max(1, total_examples) for k in target_mix
                }
                rel_frac = {
                    k: realized_counts[k] / max(1, total_examples) for k in target_mix
                }
                l1_gap = sum(abs(req_frac[k] - rel_frac[k]) for k in target_mix)

                if getattr(self.config, "report_sampler_mix", False):
                    phases = ["early", "mid", "late"]
                    outcomes = ["win", "draw", "loss"]

                    print("\n[Sampler epoch mix] requested vs realized (counts)")
                    print("bucket".ljust(18) + "req".rjust(8) + "real".rjust(8))
                    for o in outcomes:
                        for p in phases:
                            k = BucketKey(o, p)
                            print(
                                f"{o}:{p}".ljust(18)
                                + f"{requested_counts[k]:8d}{realized_counts[k]:8d}"
                            )
                    print(f"[Sampler] L1 gap (fractions): {l1_gap:.3f}\n")

            if avg_v_loss < self.best_value_loss:
                self.best_value_loss = avg_v_loss
                self.best_epoch = epoch
                self.save_checkpoint(
                    epoch=epoch, label="best", best_value_loss=avg_v_loss
                )
                print(
                    f"Best model updated (value loss = {avg_v_loss:.4f}) → saved as best"
                )

            self.save_checkpoint(epoch=epoch, label="latest")

            self.writer.add_scalar("Loss/Policy", avg_p_loss, epoch)
            self.writer.add_scalar("Loss/Value", avg_v_loss, epoch)

            if epoch % self.eval_every == 0:
                from cli.eval.eval import evaluate_model_vs_pure_mcts

                win_rate = evaluate_model_vs_pure_mcts(
                    model=self.model,
                    device=self.device,
                    config=self.config,
                    num_games=self.eval_num_games,
                    board_size=8,
                    writer=self.writer,
                    global_step=epoch,
                )

                print(
                    f"[Eval] Win Rate vs Pure MCTS after Epoch {epoch}: {win_rate:.2f}"
                )

                if win_rate >= self.target_win_rate:
                    print(
                        f"Early stopping: Target win rate {self.target_win_rate:.2f} achieved!"
                    )
                    break

        print(
            f"\nTraining complete. Best value loss = {self.best_value_loss:.4f} at epoch {self.best_epoch}"
        )
        self.writer.close()
        return self.best_epoch, self.best_value_loss

    def save_checkpoint(
        self, epoch: int, label: str = "latest", best_value_loss: Optional[float] = None
    ):
        ckpt = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": epoch,
        }
        if best_value_loss is not None:
            ckpt["best_value_loss"] = best_value_loss

        # Always write a rolling "last"
        last_path = self.save_paths.get(
            "model_last", "checkpoints/policy_value_net_last.pth"
        )
        torch.save(ckpt, last_path)

        # If it's a "best" event, also write best
        if label == "best":
            best_path = self.save_paths.get(
                "model_best", "checkpoints/policy_value_net_best.pth"
            )
            torch.save(ckpt, best_path)


def run_training(
    model: PolicyValueNet,
    optimizer: optim.Optimizer,
    buffer: ReplayBuffer,
    config: Any,
    best_value_loss: float = float("inf"),
    debug: bool = False,
    save_paths: Optional[Dict[str, Path]] = None,
) -> Tuple[Optional[int], float]:
    """Utility to create a  and start training."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    trainer = AlphaZeroTrainer(
        model, optimizer, buffer, config, device, best_value_loss, save_paths=save_paths
    )
    return trainer.train(debug=debug)

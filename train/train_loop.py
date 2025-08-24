import os
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from model.policy_value_net import PolicyValueNet
from train.replay_buffer import ReplayBuffer


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

    def _write_debug_log(self, path: Optional[str], content: str) -> None:
        """Helper method to write debug logs safely."""
        if path:
            try:
                with open(path, "a") as f:
                    f.write(content)
            except Exception as e:
                print(f"[Debug Log Error] Could not write to {path}: {e}")

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
            epoch_policy_loss = 0
            epoch_value_loss = 0
            value_preds_this_epoch = []

            if epoch % self.reload_buffer_every == 0 and epoch != 1:
                print(
                    f"[Trainer] Reloading replay buffer from disk at epoch {epoch}..."
                )
                self.replay_buffer = ReplayBuffer.load("checkpoints/replay_buffer.pkl")

            for step in range(self.steps_per_epoch):
                states, target_policies, target_values = self.replay_buffer.sample(
                    self.batch_size
                )

                states = states.to(self.device)
                target_policies = target_policies.to(self.device)
                target_values = target_values.to(torch.float32).to(self.device)

                # If legacy {0,1,2} labels sneak in, remap to {-1,0,1}
                uniq = torch.unique(target_values).tolist()
                if all(u in (0.0, 1.0, 2.0) for u in uniq):
                    target_values = target_values - 1.0

                action_size = self.model.policy_fc.out_features
                target_policies = target_policies.view(-1, action_size)

                self.optimizer.zero_grad()
                logits, value_pred = self.model(states)

                if debug:
                    with torch.no_grad():
                        value_mean = value_pred.mean().item()
                        value_std = value_pred.std().item()
                        value_preds_this_epoch.append((value_mean, value_std))

                    self._write_debug_log(
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
                    self._write_debug_log(
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
                self._write_debug_log(
                    loss_log_path,
                    f"Epoch {epoch}: Policy Loss = {avg_p_loss:.4f}, Value Loss = {avg_v_loss:.4f}\n",
                )

                mean_of_means = sum(x[0] for x in value_preds_this_epoch) / len(
                    value_preds_this_epoch
                )
                mean_of_stds = sum(x[1] for x in value_preds_this_epoch) / len(
                    value_preds_this_epoch
                )
                self._write_debug_log(
                    stats_log_path,
                    f"Epoch {epoch}: value_pred mean = {mean_of_means:.4f}, std = {mean_of_stds:.4f}\n",
                )

            print(
                f"Epoch {epoch}: Policy Loss = {avg_p_loss:.4f}, Value Loss = {avg_v_loss:.4f}"
            )

            if avg_v_loss < self.best_value_loss:
                self.best_value_loss = avg_v_loss
                self.best_epoch = epoch
                self.save_checkpoint(
                    epoch=epoch, label="best", best_value_loss=avg_v_loss
                )
                print(
                    f"Best model updated (value loss = {avg_v_loss:.4f}) → saved as best"
                )

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
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": epoch,
        }
        if best_value_loss is not None:
            checkpoint["best_value_loss"] = best_value_loss

        torch.save(checkpoint, f"checkpoints/policy_value_net_{label}.pth")


def run_training(
    model: PolicyValueNet, optimizer: optim.Optimizer, buffer: ReplayBuffer, config: Any
) -> None:
    """Utility to create a trainer and start training."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    trainer = AlphaZeroTrainer(model, optimizer, buffer, config, device)
    trainer.train()

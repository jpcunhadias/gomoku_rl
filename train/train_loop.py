import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from typing import Any, Tuple, Optional

from train.replay_buffer import ReplayBuffer


class AlphaZeroTrainer:
    """Trainer implementing the AlphaZero learning loop."""

    def __init__(
        self,
        model: nn.Module,
        replay_buffer: ReplayBuffer,
        config: Any,
        device: str = "cpu",
        start_epoch: int = 1,
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
        self.replay_buffer = replay_buffer
        self.device = device
        self.start_epoch = start_epoch
        self.batch_size = config.batch_size
        self.epochs = config.epochs
        self.steps_per_epoch = config.steps_per_epoch
        self.save_path = config.save_path

        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.policy_loss_fn = nn.KLDivLoss(reduction="batchmean")
        # Three-class classification: loss, draw, win
        self.value_loss_fn = nn.CrossEntropyLoss()
        self.best_value_loss = float("inf")
        self.best_epoch = None

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
            value_pred (Tensor): Logits from the 3-class value head.
            target_value (Tensor): Game result encoded as 0=loss, 1=draw, 2=win.
        """
        log_probs = F.log_softmax(policy_logits, dim=1)
        policy_loss = self.policy_loss_fn(log_probs, target_policy)
        value_loss = self.value_loss_fn(value_pred, target_value.view(-1))

        # total_loss = policy_loss + value_loss
        total_loss = policy_loss * 0.1 + value_loss * 1.0

        return total_loss, policy_loss.item(), value_loss.item()

    def train(self, debug: bool = False) -> Tuple[Optional[int], float]:
        """
        Main training loop.
        """
        self.model.train()

        # Create log files if debugging
        if debug:
            loss_log_path = os.path.join("checkpoints", "train_loss_summary.log")
            stats_log_path = os.path.join("checkpoints", "value_pred_stats.log")
            grad_log_path = os.path.join("checkpoints", "gradient_debug.log")
            value_debug_path = os.path.join("checkpoints", "value_pred_debug.log")

        for epoch in range(self.start_epoch, self.start_epoch + self.epochs):
            epoch_policy_loss = 0
            epoch_value_loss = 0
            value_preds_this_epoch = []

            # === Reload buffer every N epochs
            if epoch % self.reload_buffer_every == 0 and epoch != self.start_epoch:
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
                target_values = target_values.to(self.device)

                action_size = self.model.policy_fc.out_features
                target_policies = target_policies.view(-1, action_size)

                self.optimizer.zero_grad()
                logits, value_pred = self.model(states)

                # Collect stats
                if debug:
                    with torch.no_grad():
                        probs = torch.softmax(value_pred, dim=1)
                        expected = probs[:, 2] - probs[:, 0]
                        value_mean = expected.mean().item()
                        value_std = expected.std().item()
                        value_preds_this_epoch.append((value_mean, value_std))

                    # Log value head predictions (first 8 examples only)
                    with open(value_debug_path, "a") as f:
                        f.write(f"\nEpoch {epoch}, Step {step}:\n")
                        probs = torch.softmax(value_pred.detach().cpu(), dim=1)
                        expected = probs[:, 2] - probs[:, 0]
                        preds = expected.tolist()
                        targets = target_values.detach().cpu().tolist()
                        for pred, target in zip(preds[:8], targets[:8]):
                            f.write(
                                f"  z: {target}, v: {float(pred):.4f}\n"
                            )

                loss, p_loss, v_loss = self.compute_loss(
                    logits, target_policies, value_pred, target_values
                )
                loss.backward()

                # Gradient norms
                if debug:
                    with open(grad_log_path, "a") as f:
                        f.write(f"\nEpoch {epoch}, Step {step}:\n")
                        for name, param in self.model.named_parameters():
                            if param.grad is not None:
                                norm = param.grad.norm().item()
                                if "value" in name:
                                    f.write(f"[Value]  {name}: {norm:.6f}\n")
                                elif "policy" in name:
                                    f.write(f"[Policy] {name}: {norm:.6f}\n")

                self.optimizer.step()

                epoch_policy_loss += p_loss
                epoch_value_loss += v_loss

            avg_p_loss = epoch_policy_loss / self.steps_per_epoch
            avg_v_loss = epoch_value_loss / self.steps_per_epoch

            if debug:
                with open(loss_log_path, "a") as f:
                    f.write(
                        f"Epoch {epoch}: Policy Loss = {avg_p_loss:.4f}, Value Loss = {avg_v_loss:.4f}\n"
                    )

                # Aggregate value stats for this epoch
                mean_of_means = sum(x[0] for x in value_preds_this_epoch) / len(
                    value_preds_this_epoch
                )
                mean_of_stds = sum(x[1] for x in value_preds_this_epoch) / len(
                    value_preds_this_epoch
                )
                with open(stats_log_path, "a") as f:
                    f.write(
                        f"Epoch {epoch}: value_pred mean = {mean_of_means:.4f}, std = {mean_of_stds:.4f}\n"
                    )

            print(
                f"Epoch {epoch}: Policy Loss = {avg_p_loss:.4f}, Value Loss = {avg_v_loss:.4f}"
            )

            if avg_v_loss < self.best_value_loss:
                self.best_value_loss = avg_v_loss
                self.best_epoch = epoch
                self.save_checkpoint(epoch=epoch, label="best")  # Save best model
                print(
                    f"Best model updated (value loss = {avg_v_loss:.4f}) → saved as best"
                )

            self.writer.add_scalar("Loss/Policy", avg_p_loss, epoch)
            self.writer.add_scalar("Loss/Value", avg_v_loss, epoch)

            # === Evaluate model against Pure MCTS after training ===
            if epoch % self.eval_every == 0:
                from cli.eval.eval import evaluate_model_vs_pure_mcts

                win_rate = evaluate_model_vs_pure_mcts(
                    model=self.model,
                    device=self.device,
                    num_games=self.eval_num_games,
                    board_size=8,
                    num_simulations=self.eval_num_simulations,
                    writer=self.writer,
                    global_step=epoch,
                )

                print(
                    f"[Eval] Win Rate vs Pure MCTS after Epoch {epoch}: {win_rate:.2f}"
                )

                # Early stopping if we reach target win rate
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
        self, epoch: Optional[int] = None, label: Optional[str] = None
    ) -> None:
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": epoch,  # Real integer epoch saved
        }
        if label is not None:
            path = self.save_path.replace(".pth", f"_{label}.pth")  # e.g., best
        elif epoch is not None:
            path = self.save_path.replace(".pth", f"_epoch_{epoch}.pth")
        else:
            path = self.save_path
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to: {path}")


def run_training(model: nn.Module, buffer: ReplayBuffer, config: Any) -> None:
    """Utility to create a trainer and start training."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    trainer = AlphaZeroTrainer(model, buffer, config, device)
    trainer.train()

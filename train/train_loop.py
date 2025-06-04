import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


class AlphaZeroTrainer:
    def __init__(self, model, replay_buffer, config, device="cpu"):
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
        self.batch_size = config.batch_size
        self.epochs = config.epochs
        self.steps_per_epoch = config.steps_per_epoch
        self.save_path = config.save_path

        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.policy_loss_fn = nn.KLDivLoss(reduction="batchmean")
        self.value_loss_fn = nn.BCEWithLogitsLoss()
        self.best_value_loss = float("inf")
        self.best_epoch = None

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

    def compute_loss(self, policy_logits, target_policy, value_pred, target_value):
        """
        Combines policy and value losses.

        Args:
            policy_logits (Tensor): Raw logits from policy head.
            target_policy (Tensor): Target probabilities (π from MCTS).
            value_pred (Tensor): Scalar value prediction from value head.
            target_value (Tensor): Actual game result (-1, 0, 1).
        """
        log_probs = F.log_softmax(policy_logits, dim=1)
        policy_loss = self.policy_loss_fn(log_probs, target_policy)
        value_loss = self.value_loss_fn(
            value_pred.view(-1), target_value.float().view(-1)
        )

        # total_loss = policy_loss + value_loss
        total_loss = policy_loss * 0.1 + value_loss * 1.0

        return total_loss, policy_loss.item(), value_loss.item()

    def train(self, debug=False):
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

        for epoch in range(1, self.epochs + 1):
            epoch_policy_loss = 0
            epoch_value_loss = 0
            value_preds_this_epoch = []

            for step in range(self.steps_per_epoch):
                states, target_policies, target_values = self.replay_buffer.sample(
                    self.batch_size
                )

                states = states.to(self.device)
                target_policies = target_policies.to(self.device)
                target_values = ((target_values + 1) / 2).to(self.device)

                target_policies = target_policies.view(-1, 225)

                self.optimizer.zero_grad()
                logits, value_pred = self.model(states)

                # Collect stats
                if debug:
                    with torch.no_grad():
                        value_mean = value_pred.mean().item()
                        value_std = value_pred.std().item()
                        value_preds_this_epoch.append((value_mean, value_std))

                    # Log value head predictions (first 8 examples only)
                    with open(value_debug_path, "a") as f:
                        f.write(f"\nEpoch {epoch}, Step {step}:\n")
                        preds = value_pred.detach().cpu().squeeze().tolist()
                        targets = target_values.detach().cpu().squeeze().tolist()
                        for pred, target in zip(preds[:8], targets[:8]):
                            pred_val = (
                                pred[0] if isinstance(pred, (list, tuple)) else pred
                            )
                            target_val = (
                                target[0]
                                if isinstance(target, (list, tuple))
                                else target
                            )
                            f.write(
                                f"  z: {float(target_val):.4f}, v: {float(pred_val):.4f}\n"
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
                self.save_checkpoint(epoch="best")
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
                    board_size=15,
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

    def save_checkpoint(self, epoch=None):
        """
        Saves the model and optimizer state dicts.
        """
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": epoch,
        }
        path = (
            self.save_path
            if epoch is None
            else self.save_path.replace(".pth", f"_epoch{epoch}.pth")
        )
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to: {path}")


def run_training(model, buffer, config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    trainer = AlphaZeroTrainer(model, buffer, config, device)
    trainer.train()

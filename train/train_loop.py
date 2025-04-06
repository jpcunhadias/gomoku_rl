import os
import torch
import torch.nn as nn
import torch.optim as optim


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
        self.value_loss_fn = nn.MSELoss()

    def compute_loss(self, policy_logits, target_policy, value_pred, target_value):
        """
        Combines policy and value losses.

        Args:
            policy_logits (Tensor): Raw logits from policy head.
            target_policy (Tensor): Target probabilities (π from MCTS).
            value_pred (Tensor): Scalar value prediction from value head.
            target_value (Tensor): Actual game result (-1, 0, 1).
        """
        policy_loss = self.policy_loss_fn(policy_logits, target_policy)
        value_loss = self.value_loss_fn(value_pred.squeeze(), target_value)
        total_loss = policy_loss + value_loss
        return total_loss, policy_loss.item(), value_loss.item()

    def train(self):
        """
        Main training loop.
        """
        self.model.train()

        for epoch in range(1, self.epochs + 1):
            epoch_policy_loss = 0
            epoch_value_loss = 0

            for step in range(self.steps_per_epoch):
                states, target_policies, target_values = self.replay_buffer.sample(
                    self.batch_size
                )

                states = states.to(self.device)
                target_policies = target_policies.to(self.device)
                target_values = target_values.to(self.device)

                target_policies = target_policies.view(-1, 225)

                self.optimizer.zero_grad()
                logits, value_pred = self.model(states)
                loss, p_loss, v_loss = self.compute_loss(
                    logits, target_policies, value_pred, target_values
                )
                loss.backward()
                self.optimizer.step()

                epoch_policy_loss += p_loss
                epoch_value_loss += v_loss

            avg_p_loss = epoch_policy_loss / self.steps_per_epoch
            avg_v_loss = epoch_value_loss / self.steps_per_epoch
            print(
                f"Epoch {epoch}: Policy Loss = {avg_p_loss:.4f}, Value Loss = {avg_v_loss:.4f}"
            )

            self.save_checkpoint(epoch)

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

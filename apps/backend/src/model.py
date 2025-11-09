import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import os
import datetime
from typing import Any

class LinearQNet(nn.Module):
    """
    A simple neural network for Q-learning in the Snake game.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()

        # Two fully connected layers
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the neural network.
        """
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, folder: str = "model") -> None:
        """
        Save the trained model to disk with a timestamp.

        Args:
            folder: The directory where the model file should be saved.
        """
        # Ensure save directory exists
        if not os.path.exists(folder):
            os.makedirs(folder)

        # Generate unique timestamped filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        file_name = os.path.join(folder, f"model_{timestamp}.pth")

        # Save model weights
        torch.save(self.state_dict(), file_name)

        print(f"[LinearQNet] Model saved as {file_name}")

    def load(self, file_name: str) -> None:
        """
        Load a previously saved model from disk.

        Args:
            file_name: Path to the saved .pth file.
        """
        if not os.path.exists(file_name):
            raise FileNotFoundError(f"[LinearQNet] Model file not found: {file_name}")

        self.load_state_dict(torch.load(file_name))
        print(f"[LinearQNet] Model loaded from {file_name}")


class QTrainer:
    """
    Trainer class for the Q-learning neural network.

    Handles the training process using the Bellman equation:
    Q(s,a) = r + γ * max(Q(s',a'))

    Where:
    - Q(s,a) = Q-value for state s and action a
    - r = immediate reward
    - γ = discount factor (gamma)
    - s' = next state
    - a' = possible actions in next state
    """

    def __init__(self, model: Any, lr: float, gamma: float) -> None:
        """
        Initialize the trainer with model and hyperparameters.

        Args:
            model: The neural network to train
            lr: Learning rate for the optimizer
            gamma: Discount factor for future rewards
        """
        self.model = model
        self.lr = lr
        self.gamma = gamma

        # Adam optimizer (works better for RL than plain SGD)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # Mean squared error loss (used for Q-value regression)
        self.criterion = nn.MSELoss()

        pass

    def train_step(self, state, action, reward, next_state, done):
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = torch.tensor([done], dtype=torch.bool)

        # Predict current Q-values
        pred = self.model(state)

        # Clone without gradient tracking
        target = pred.clone().detach()

        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                # Use detached next_state prediction (important!)
                next_q = torch.max(self.model(next_state[idx]).detach())
                Q_new = reward[idx] + self.gamma * next_q

            # Only update the action that was actually taken
            target[idx][action[idx].item()] = Q_new

        # Standard backpropagation
        self.optimizer.zero_grad()
        loss = self.criterion(pred, target)
        loss.backward()
        self.optimizer.step()

        # Optional debug info
        avg_q = torch.mean(torch.max(pred, dim=1)[0]).item()
        print(f"[train] loss={loss.item():.3f} avg_Q={avg_q:.3f}")
import torch
import random
import numpy as np
from typing import List
from collections import deque
from model import LinearQNet, QTrainer

MAX_MEMORY = 100_000  # max replay buffer size
BATCH_SIZE = 256    # how many experiences to train on at once
LR = 0.001            # learning rates

class DQN:
    def __init__(self):
        """
        Initialize the DQN agent.
        """
        self.n_games = 0
        self.epsilon = 40  # controls randomness
        self.gamma = 0.92

        # Experience replay memory
        self.memory = deque(maxlen=MAX_MEMORY)

        # Neural network and optimizer
        self.model = LinearQNet(13, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game: "Game") -> List[float]:
        snake = game.snake
        head_x, head_y = snake.head
        food_x, food_y = game.food.position
        dx, dy = snake.direction
        grid_w, grid_h = game.grid_width, game.grid_height

        def danger_at(pos):
            x, y = pos
            if x < 0 or x >= grid_w or y < 0 or y >= grid_h:
                return True
            if (x, y) in snake.body[1:]:
                return True
            return False

        directions = {
            "UP": (0, -1),
            "RIGHT": (1, 0),
            "DOWN": (0, 1),
            "LEFT": (-1, 0),
        }

        if snake.direction == directions["UP"]:
            current_dir = "UP"
            left_dir = "LEFT"
            right_dir = "RIGHT"
        elif snake.direction == directions["DOWN"]:
            current_dir = "DOWN"
            left_dir = "RIGHT"
            right_dir = "LEFT"
        elif snake.direction == directions["LEFT"]:
            current_dir = "LEFT"
            left_dir = "DOWN"
            right_dir = "UP"
        else:
            current_dir = "RIGHT"
            left_dir = "UP"
            right_dir = "DOWN"

        straight_vec = directions[current_dir]
        left_vec = directions[left_dir]
        right_vec = directions[right_dir]

        danger_straight = danger_at((head_x + straight_vec[0], head_y + straight_vec[1]))
        danger_right = danger_at((head_x + right_vec[0], head_y + right_vec[1]))
        danger_left = danger_at((head_x + left_vec[0], head_y + left_vec[1]))

        dir_up = current_dir == "UP"
        dir_down = current_dir == "DOWN"
        dir_left = current_dir == "LEFT"
        dir_right = current_dir == "RIGHT"

        food_up = food_y > head_y
        food_down = food_y < head_y
        food_left = food_x < head_x
        food_right = food_x > head_x

        dist_x = (food_x - head_x) / grid_w
        dist_y = (food_y - head_y) / grid_h

        state = [
            int(danger_straight),
            int(danger_right),
            int(danger_left),
            int(dir_up),
            int(dir_down),
            int(dir_left),
            int(dir_right),
            int(food_up),
            int(food_down),
            int(food_left),
            int(food_right),
            dist_x,
            dist_y,
        ]
        return state

    def calculate_reward(self, game: "Game", done: bool) -> float:
        """Reward shaping that actually teaches food-seeking."""
        snake = game.snake
        head_x, head_y = snake.head
        food_x, food_y = game.food.position

        new_distance = abs(head_x - food_x) + abs(head_y - food_y)
        prev_distance = getattr(self, "prev_distance", None)

        reward = 0.0

        # Reward for eating food
        if game.score > getattr(self, "prev_score", 0):
            reward += 10

        # Reward for surviving
        reward += 0.1

        # Encourage moving closer
        if prev_distance is not None:
            if new_distance < prev_distance:
                reward += 1.0
            else:
                reward -= 1.0

        # Penalty for dying
        if done:
            reward -= 10

        # Save for next iteration
        self.prev_distance = new_distance
        self.prev_score = game.score
        return reward

    def remember(self, state, action, reward, next_state, done):
        """
        Store an experience tuple in replay memory.
        """
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self) -> None:
        if len(self.memory) == 0:
            return

        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)

        actions_tensor = torch.tensor(actions, dtype=torch.long)

        self.trainer.train_step(
            torch.tensor(states, dtype=torch.float),
            actions_tensor,
            torch.tensor(rewards, dtype=torch.float),
            torch.tensor(next_states, dtype=torch.float),
            torch.tensor(dones, dtype=torch.bool),
        )

    def train_short_memory(self, state, action, reward, next_state, done) -> None:

        self.trainer.train_step(
            torch.tensor(state, dtype=torch.float),
            torch.tensor(action, dtype=torch.long),
            torch.tensor(reward, dtype=torch.float),
            torch.tensor(next_state, dtype=torch.float),
            done,
        )

    def get_action(self, state: List[float]) -> List[int]:
        self.epsilon = max(10, 80 - (self.n_games // 2))
        final_move = [0, 0, 0]

        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)  # explore
        else:
            state_tensor = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state_tensor)
            move = torch.argmax(prediction).item()  # exploit

        final_move[move] = 1
        return final_move

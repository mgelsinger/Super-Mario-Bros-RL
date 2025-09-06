# agent.py — GPU-friendly replay + minimal transfers

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage

from agent_nn import AgentNN


class Agent:
    def __init__(self, input_dims, num_actions):
        # --- core settings (tune as desired) ---
        self.gamma = 0.99
        self.lr = 2.5e-4
        self.batch_size = 64              # keep modest to reduce transfer cost
        self.buffer_size = 100_000

        # --- device / nets ---
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.benchmark = True

        self.online_network = AgentNN(input_dims, num_actions).to(self.device)
        self.target_network = AgentNN(input_dims, num_actions).to(self.device)
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(self.online_network.parameters(), lr=self.lr)

        # --- replay buffer on CPU (compact), sampling to GPU per-tensor ---
        # States will be stored as uint8 (0..255) with shape (4,84,84)
        self.replay_buffer = TensorDictReplayBuffer(
            storage=LazyMemmapStorage(self.buffer_size),  # CPU, memory-mapped
            batch_size=None,
        )

        self.num_actions = num_actions
        self.learn_steps = 0
        self.target_update_interval = 5_000

    def choose_action(self, state, epsilon: float = 0.05):
        # state: numpy (4,84,84) uint8
        if torch.rand(1).item() < epsilon:
            return torch.randint(0, self.num_actions, ()).item()
        with torch.no_grad():
            s = torch.from_numpy(state).to(self.device, dtype=torch.float32, non_blocking=True) / 255.0
            s = s.unsqueeze(0)  # (1,C,H,W)
            q = self.online_network(s)
            return int(torch.argmax(q, dim=1).item())

    def store_in_memory(self, state, action, reward, next_state, done):
        """
        state / next_state: numpy uint8 (4,84,84)
        action: int
        reward: float
        done: bool
        """
        td = TensorDict(
            {
                "state": torch.from_numpy(state).to(torch.uint8),        # CPU uint8
                "action": torch.tensor(action, dtype=torch.int64),
                "reward": torch.tensor(reward, dtype=torch.float32),
                "next_state": torch.from_numpy(next_state).to(torch.uint8),
                "done": torch.tensor(done, dtype=torch.bool),
            },
            batch_size=[],
        )
        self.replay_buffer.add(td)

    def learn(self):
        # Don’t try to learn without enough samples
        if len(self.replay_buffer) < self.batch_size * 10:
            return

        # Sample a small batch on CPU; then move ONLY what we need
        batch = self.replay_buffer.sample(self.batch_size)

        # Move / cast explicitly, not via batch.to(...)
        s = batch["state"].to(self.device, dtype=torch.float32, non_blocking=True) / 255.0  # (B,4,84,84)
        ns = batch["next_state"].to(self.device, dtype=torch.float32, non_blocking=True) / 255.0
        a = batch["action"].to(self.device, dtype=torch.int64, non_blocking=True).view(-1)
        r = batch["reward"].to(self.device, dtype=torch.float32, non_blocking=True).view(-1)
        d = batch["done"].to(self.device, dtype=torch.bool, non_blocking=True).view(-1)

        # Q-learning target
        with torch.no_grad():
            q_next = self.target_network(ns).max(dim=1).values
            target = r + (1.0 - d.float()) * self.gamma * q_next

        q_pred = self.online_network(s).gather(1, a.unsqueeze(1)).squeeze(1)

        loss = F.smooth_l1_loss(q_pred, target)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_network.parameters(), 10.0)
        self.optimizer.step()

        self.learn_steps += 1
        if self.learn_steps % self.target_update_interval == 0:
            self.target_network.load_state_dict(self.online_network.state_dict())

import torch
import numpy as np

class PrioritizedReplayBuffer:
    def __init__(self, capacity, state_shape, alpha=0.6, beta_start=0.4, beta_frames=1_000_000, device='cuda'):
        self.capacity = capacity
        self.device = device
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames

        # GPU tensors directly (no NumPy)
        self.states = torch.zeros((capacity, *state_shape), dtype=torch.float32, device=device)
        self.actions = torch.zeros((capacity, 1), dtype=torch.int64, device=device)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.next_states = torch.zeros((capacity, *state_shape), dtype=torch.float32, device=device)
        self.dones = torch.zeros((capacity, 1), dtype=torch.float32, device=device)

        # Priorities still on CPU (sampling faster on CPU)
        self.priorities = np.zeros((capacity,), dtype=np.float32)

        self.ptr = 0
        self.size = 0
        self.frame = 1

    def push(self, state, action, reward, next_state, done):
        """Push data directly to GPU buffers."""
        max_prio = self.priorities.max() if self.size > 0 else 1.0

        # Convert to tensors once
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device)
        next_state_t = torch.tensor(next_state, dtype=torch.float32, device=self.device)

        self.states[self.ptr] = state_t
        self.actions[self.ptr] = torch.tensor(action, dtype=torch.int64, device=self.device)
        self.rewards[self.ptr] = torch.tensor(reward, dtype=torch.float32, device=self.device)
        self.next_states[self.ptr] = next_state_t
        self.dones[self.ptr] = torch.tensor(done, dtype=torch.float32, device=self.device)
        self.priorities[self.ptr] = max_prio

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        self.frame += 1
        prios = self.priorities[:self.size]
        probs = prios ** self.alpha
        probs = probs / probs.sum() if probs.sum() != 0 else np.ones_like(probs) / len(probs)

        indices = np.random.choice(self.size, batch_size, p=probs)
        beta = self._beta_by_frame()

        weights = (self.size * probs[indices]) ** (-beta)
        weights /= weights.max()

        # Convert everything to GPU tensors (already there)
        weights_t = torch.tensor(weights, dtype=torch.float32, device=self.device)
        indices_t = torch.tensor(indices, dtype=torch.long, device=self.device)

        return (
            self.states[indices_t],
            self.actions[indices_t],
            self.rewards[indices_t],
            self.next_states[indices_t],
            self.dones[indices_t],
            weights_t,
            indices_t,
        )

    def update_priorities(self, indices, td_errors):
        new_prios = td_errors.detach().abs().cpu().numpy() + 1e-6
        self.priorities[indices.cpu().numpy()] = new_prios

    def _beta_by_frame(self):
        return min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)

    def __len__(self):
        return self.size
import torch
import numpy as np

class PrioritizedReplayBuffer:
    def __init__(self, capacity, state_shape, alpha=0.6, beta_start=0.4, beta_frames=1_000_000, device='cuda'):
        self.capacity = capacity
        self.device = device
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames

        # Main data (GPU tensors)
        self.states = torch.zeros((capacity, *state_shape), dtype=torch.float32, device=device)
        self.actions = torch.zeros((capacity, 1), dtype=torch.int64, device=device)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.next_states = torch.zeros((capacity, *state_shape), dtype=torch.float32, device=device)
        self.dones = torch.zeros((capacity, 1), dtype=torch.float32, device=device)

        # Priorities on CPU
        self.priorities = np.zeros((capacity,), dtype=np.float32)

        self.ptr = 0
        self.size = 0
        self.frame = 1

    def push(self, state, action, reward, next_state, done):
        max_prio = max(self.priorities.max() if self.size > 0 else 1.0, 1e-6)

        if not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
        if not torch.is_tensor(next_state):
            next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)

        self.states[self.ptr] = state
        self.actions[self.ptr] = torch.tensor(action, dtype=torch.int64, device=self.device)
        self.rewards[self.ptr] = torch.tensor(reward, dtype=torch.float32, device=self.device)
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = torch.tensor(done, dtype=torch.float32, device=self.device)
        self.priorities[self.ptr] = max_prio

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        self.frame += 1
        prios = self.priorities[:self.size]
        prios = np.nan_to_num(prios, nan=1.0, posinf=1.0, neginf=1e-6)
        probs = prios ** self.alpha
        probs /= probs.sum() if probs.sum() != 0 else len(prios)

        indices = np.random.choice(self.size, batch_size, p=probs)
        beta = self._beta_by_frame()

        weights = (1.0 / (self.size * probs[indices])) ** beta
        weights /= weights.max()

        indices_t = torch.tensor(indices, dtype=torch.long, device=self.device)
        weights_t = torch.tensor(weights, dtype=torch.float32, device=self.device)

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
        if isinstance(td_errors, torch.Tensor):
            td_errors = td_errors.detach().abs().cpu().numpy()
        else:
            td_errors = np.abs(td_errors)
        td_errors = np.squeeze(td_errors) + 1e-6

        if isinstance(indices, torch.Tensor):
            indices = indices.cpu().numpy()

        self.priorities[indices] = td_errors

    def _beta_by_frame(self):
        return min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)

    def __len__(self):
        return self.size

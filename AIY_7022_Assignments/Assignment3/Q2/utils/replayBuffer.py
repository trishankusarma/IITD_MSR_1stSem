import numpy as np
import torch
from collections import namedtuple
from collections import deque

# Simple Bounded Replay Buffer
class ReplayBuffer:
    def __init__(self, max_size=100_000):
        self.max_size = max_size
        self._buf = deque(maxlen=max_size)

    def push(self, s, a, r, ns, na, done):
        # store as numpy arrays for compactness
        self._buf.append((np.array(s, copy=False),
                          np.int64(a),
                          np.float32(r),
                          np.array(ns, copy=False),
                          np.int64(na),
                          np.float32(done)))

    def sample(self, batch_size):
        idx = np.random.choice(len(self._buf), size=batch_size, replace=False)
        states, actions, rewards, next_states, next_actions, dones = zip(*(self._buf[i] for i in idx))
        states = torch.tensor(np.stack(states), dtype=torch.float32)
        actions = torch.tensor(np.array(actions), dtype=torch.long)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
        next_states = torch.tensor(np.stack(next_states), dtype=torch.float32)
        next_actions = torch.tensor(np.array(next_actions), dtype=torch.long)
        dones = torch.tensor(np.array(dones), dtype=torch.float32)
        return states, actions, rewards, next_states, next_actions, dones

    def __len__(self):
        return len(self._buf)

    def clear(self):
        self._buf.clear()
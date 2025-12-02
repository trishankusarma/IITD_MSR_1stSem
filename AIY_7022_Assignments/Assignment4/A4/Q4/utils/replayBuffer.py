import torch
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity, state_shape, device=None):
        """
        Efficient Uniform Experience Replay Buffer
        Stores entire buffer on GPU for fast sampling.

        Args:
            capacity: max number of transitions
            state_shape: shape of each state (e.g., (8,))
            device: torch device (GPU recommended)
        """
        self.capacity = int(capacity)
        self.device = device if device else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

        # Allocate memory once — no dynamic resizing
        self.states = torch.zeros((self.capacity, *state_shape), 
                                  dtype=torch.float32, device=self.device)
        self.actions = torch.zeros((self.capacity, 1), 
                                   dtype=torch.long, device=self.device)
        self.rewards = torch.zeros((self.capacity, 1), 
                                   dtype=torch.float32, device=self.device)
        self.next_states = torch.zeros((self.capacity, *state_shape),
                                       dtype=torch.float32, device=self.device)
        self.dones = torch.zeros((self.capacity, 1),
                                 dtype=torch.float32, device=self.device)

        self.ptr = 0
        self.size = 0
    
    def push(self, state, action, reward, next_state, done):
        """
        Store a transition in the replay buffer.
        """

        # Convert numpy → tensor only when needed
        self.states[self.ptr].copy_(
            torch.as_tensor(state, dtype=torch.float32, device=self.device)
        )
        self.actions[self.ptr] = int(action)
        self.rewards[self.ptr] = float(reward)

        self.next_states[self.ptr].copy_(
            torch.as_tensor(next_state, dtype=torch.float32, device=self.device)
        )
        self.dones[self.ptr] = float(done)

        # Move pointer
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """
        Sample a random batch of transitions uniformly.
        Fully on GPU.
        """
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )

    def __len__(self):
        return self.size

    def reset(self):
        """Useful for debugging or resetting training."""
        self.ptr = 0
        self.size = 0
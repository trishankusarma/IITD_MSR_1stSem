import torch
import torch.nn.functional as F
import numpy as np
import math, random
from utils.replayBuffer import PrioritizedReplayBuffer
from utils.networks import DQNNet
from utils.actions import ALL_ACTIONS

class DQNAgent:
    def __init__(self, n_obs, num_actions, device,
                 buffer_size=1_000_000, batch_size=64,
                 gamma=0.99, lr=1e-4, target_update_freq=1000,
                 eps_start=1.0, eps_end=0.05, eps_tau=120_000,
                 alpha=0.6, beta_start=0.4, beta_frames=1_000_000):

        self.device = device
        self.num_actions = num_actions
        self.policy_net = DQNNet(n_obs, num_actions).to(device)
        self.target_net = DQNNet(n_obs, num_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.buffer = PrioritizedReplayBuffer(
            capacity=buffer_size,
            state_shape=(n_obs,),
            alpha=alpha,
            beta_start=beta_start,
            beta_frames=beta_frames,
            device=device
        )
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)

        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.eps_start, self.eps_end, self.eps_tau = eps_start, eps_end, eps_tau
        self.steps_done = 0
    
    def get_epsilon(self):
        return self.eps_end + (self.eps_start - self.eps_end) * math.exp(-self.steps_done / self.eps_tau)

    def select_action(self, state, eps):
        self.steps_done += 1
        if random.random() < eps:
            return random.randrange(len(ALL_ACTIONS))
        with torch.no_grad():
            s_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return self.policy_net(s_t).argmax(1).item()

    def optimize_model(self):
        if len(self.buffer) < self.batch_size:
            return None

        # Sample directly from GPU buffer
        states, actions, rewards, next_states, dones, weights, indices = self.buffer.sample(self.batch_size)

        q_values = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
            next_q_values = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards + (1.0 - dones.float()) * self.gamma * next_q_values

        td_errors = target_q - q_values
        loss = (weights * F.smooth_l1_loss(q_values, target_q, reduction='none')).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.optimizer.step()

        self.buffer.update_priorities(indices, td_errors.detach().abs().cpu().numpy() + 1e-6)
        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    # Warmup & Evaluation
    def warmup_replay_buffer(self, env, min_size=2000):
        print(f"Filling replay buffer with {min_size} random transitions...")
        while len(self.buffer) < min_size:
            state = env.reset()
            for _ in range(10):
                action_idx = random.randrange(self.num_actions)
                action = ALL_ACTIONS[action_idx]
                next_state, reward, done, _ = env.step(action)
                self.buffer.push(state, action_idx, reward, next_state, done)
                state = next_state
                if len(self.buffer) >= min_size:
                    break

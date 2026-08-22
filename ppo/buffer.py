import numpy as np
import torch

class RolloutBuffer:
    def __init__(self, buffer_size, obs_dim, action_dim):
        self.obs = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)

        self.buffer_size = buffer_size
        self.ptr = 0

    def store(self, obs, action, log_prob, value, reward, done):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr] = value
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.ptr += 1

    def is_full(self):
        return self.ptr >= self.buffer_size

    def reset(self):
        self.ptr = 0

    def get(self):
        return{
            "obs" : torch.as_tensor(self.obs),
            "actions" : torch.as_tensor(self.actions),
            "log_probs" : torch.as_tensor(self.log_probs),
            "values" : torch.as_tensor(self.values),
            "rewards" : torch.as_tensor(self.rewards),
            "dones" : torch.as_tensor(self.dones),
        }
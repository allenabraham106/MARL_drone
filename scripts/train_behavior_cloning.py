import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ppo"))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from policy import ActorCritic

observations = np.load("demo_observations.npy")
actions = np.load("demo_actions.npy")

obs_tensor = torch.as_tensor(observations, dtype=torch.float32)
actions_tensor = torch.as_tensor(actions, dtype=torch.float32)

policy = ActorCritic(obs_dim=19, action_dim=4)
optimizer = optim.Adam(policy.parameters(), lr=1e-3)

n_samples = obs_tensor.shape[0]
batch_size = 256
n_epochs = 20

for epoch in range(n_epochs):
    indices = torch.randperm(n_samples)
    total_loss = 0.0

    for start in range(0, n_samples, batch_size):
        end = start + batch_size
        batch_idx = indices[start:end]
        batch_obs = obs_tensor[batch_idx]
        batch_actions = actions_tensor[batch_idx]

        action_mean, value = policy.forward(batch_obs)

        # squashing network's raw output the same way get_action does
        thrust = torch.sigmoid(action_mean[:, 0:1]) * 6.0
        rates = torch.tanh(action_mean[:, 1:4]) * 5.0
        predicted_action = torch.cat([thrust, rates], dim=-1)

        loss = nn.functional.mse_loss(predicted_action, batch_actions)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(
        f"Epoch {epoch + 1}/{n_epochs} | loss: {total_loss / (n_samples // batch_size):.4f}"
    )

torch.save(policy.state_dict(), "checkpoints/pretrained_bc_policy.pt")
print("Saved checkpoints/pretrained_bc_policy.pt")
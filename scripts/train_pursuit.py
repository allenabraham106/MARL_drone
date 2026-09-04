import sys, os

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "envs")
)
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ppo")
)

import torch
import time
import numpy as np
from pursuit_env import PursuitEnv
from policy import ActorCritic
from buffer import RolloutBuffer
from ppo import compute_gae, PPO

env = PursuitEnv()
policy = ActorCritic(obs_dim=19, action_dim=4)
buffer = RolloutBuffer(buffer_size=2048, obs_dim=19, action_dim=4)
ppo = PPO(policy, entropy_coef=0.001)
run_id = int(time.time())

obs, info = env.reset()
n_iterations = 1000

for iteration in range(n_iterations):
    curriculum_iterations = 1000
    env.flee_scale = min(1.0, iteration / curriculum_iterations)
    buffer.reset()
    crash_count = 0
    catch_count = 0
    lost_count = 0
    timeout_count = 0

    for _ in range(buffer.buffer_size):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        action, raw_action, log_prob, value = policy.get_action(obs_tensor)

        action_np = action.detach().numpy()
        raw_action_np = raw_action.detach().numpy()

        next_obs, reward, terminated, truncated, info = env.step(action_np)
        done = terminated or truncated

        buffer.store(obs, raw_action_np, log_prob.item(), value.item(), reward, done)

        obs = next_obs
        if done:
            if terminated:
                d1_height = env.d.qpos[2]
                dist = np.linalg.norm(env.d.qpos[7:10] - env.d.qpos[0:3])
                if d1_height < 0.05:
                    crash_count += 1
                elif dist < 0.2:
                    catch_count += 1
                elif dist > 4.0:
                    lost_count += 1
            else:
                timeout_count += 1
            obs, info = env.reset()

    print("Buffer full:", buffer.is_full())
    print("Mean reward collected:", buffer.rewards.mean())

    with torch.no_grad():
        _, last_value = policy.forward(torch.as_tensor(obs, dtype=torch.float32))
    last_value = last_value.item()

    advantages, returns = compute_gae(
        buffer.rewards, buffer.values, buffer.dones, last_value
    )
    print("advantages: mean", advantages.mean(), "std", advantages.std())
    print("returns: mean", returns.mean(), "std", returns.std())

    data = buffer.get()
    advantages_t = torch.as_tensor(advantages, dtype=torch.float32)
    returns_t = torch.as_tensor(returns, dtype=torch.float32)

    ppo.update(data["obs"], data["actions"], data["log_probs"], advantages_t, returns_t)
    mean_reward = buffer.rewards.mean()
    episode_ends = np.where(buffer.dones == 1)[0]
    mean_episode_length = (
        np.mean(np.diff(np.concatenate([[-1], episode_ends])))
        if len(episode_ends) > 0
        else buffer.buffer_size
    )
    log_std_min, log_std_max = -2.0, -0.3
    log_std_display = log_std_min + 0.5 * (log_std_max - log_std_min) * (torch.tanh(policy.actor_log_std) + 1)
    current_std = torch.exp(log_std_display).detach().numpy()    
    print(
        f"Iteration {iteration:4d} | flee_scale: {env.flee_scale:.2f} | mean reward: {mean_reward:.3f} | mean episode length: {mean_episode_length:.1f} "
        f"| catches: {catch_count} | crashes: {crash_count} | lost: {lost_count} | timeouts: {timeout_count} "
        f"| action std: {current_std}"
    )
    if iteration % 100 == 0:
        torch.save(
            policy.state_dict(),
            f"checkpoints/pursuit_checkpoint_{run_id}_{iteration}.pt",
        )
print("PPO update complete")

import torch
from policy import ActorCritic


def transfer_hover_to_pursuit(hover_checkpoint_path, obs_dim=19, action_dim=4):
    hover_sd = torch.load(hover_checkpoint_path)

    pursuit_policy = ActorCritic(obs_dim=obs_dim, action_dim=action_dim)
    pursuit_sd = pursuit_policy.state_dict()

    # shared.0: input dim changed (16 -> 19), copy the overlapping columns only
    hover_w = hover_sd["shared.0.weight"]  # shape [64, 16]
    pursuit_sd["shared.0.weight"][:, : hover_w.shape[1]] = hover_w
    pursuit_sd["shared.0.bias"] = hover_sd["shared.0.bias"]

    # everything else has identical shapes -> direct copy
    for key in [
        "shared.2.weight",
        "shared.2.bias",
        "actor_mean.weight",
        "actor_mean.bias",
        "critic.weight",
        "critic.bias",
    ]:
        pursuit_sd[key] = hover_sd[key]

    pursuit_policy.load_state_dict(pursuit_sd)
    return pursuit_policy

from typing import List
import numpy as np
import torch
from torch.distributions import Categorical, Normal
from .policy import flatten_obs


class Trajectory:
    def __init__(self):
        self.obs = []
        self.next_obs = []
        self.actions = []
        self.logprobs = []
        self.entropies = []
        self.rewards = []
        self.terminated = False
        self.truncated = False

    def discounted_return(self, gamma):
        G = np.zeros_like(self.rewards[0], dtype=np.float32)
        pw = 1.0
        for r in self.rewards:
            G += pw * r
            pw *= gamma
        return G

    def sum_logprob(self):
        return torch.stack(self.logprobs).sum()

    def sum_entropy(self):
        return torch.stack(self.entropies).sum()


def _dist_with_temp(policy, obs_t, policy_kind, action_temp):
    temp = float(max(1e-06, action_temp))
    if policy_kind == "discrete" and hasattr(policy, "net"):
        logits = policy.net(obs_t)
        return Categorical(logits=logits / temp)
    if policy_kind == "continuous" and hasattr(policy, "mu_net") and hasattr(policy, "log_std"):
        mu = policy.mu_net(obs_t)
        std = torch.exp(policy.log_std) * temp
        return Normal(mu, std)
    return policy(obs_t)


def rollout_episode(
    env, policy, policy_kind, device, gamma, max_steps, action_temp=1.0, track_grad=True
):
    (obs, _) = env.reset()
    obs = flatten_obs(obs)
    traj = Trajectory()
    for _t in range(max_steps):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
        ctx = torch.enable_grad() if track_grad else torch.no_grad()
        with ctx:
            dist = _dist_with_temp(policy, obs_t, policy_kind, action_temp)
            ent = dist.entropy()
            if policy_kind == "discrete":
                a = dist.sample()
                logp = dist.log_prob(a)
                ent_step = ent
                action = int(a.item())
            else:
                a_pre = dist.sample()
                logp = dist.log_prob(a_pre).sum()
                ent_step = ent.sum()
                a_np = a_pre.detach().cpu().numpy()
                low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
                high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
                a_env = np.clip(a_np.reshape(-1), low, high)
                action = a_env
        (next_obs, reward, terminated, truncated, _info) = env.step(action)
        traj.obs.append(obs)
        traj.next_obs.append(flatten_obs(next_obs))
        if policy_kind == "discrete":
            traj.actions.append(np.array(action))
        else:
            traj.actions.append(np.array(a_np.reshape(-1), dtype=np.float32))
        if track_grad:
            traj.logprobs.append(logp)
            traj.entropies.append(ent_step)
        else:
            traj.logprobs.append(logp.detach())
            traj.entropies.append(ent_step.detach())
        traj.rewards.append(np.asarray(reward, dtype=np.float32))
        obs = flatten_obs(next_obs)
        if terminated or truncated:
            traj.terminated = bool(terminated)
            traj.truncated = bool(truncated)
            break
    return traj


def collect_trajectories(
    env, policy, policy_kind, device, gamma, max_steps, n_episodes, action_temp=1.0, track_grad=True
):
    return [
        rollout_episode(
            env,
            policy,
            policy_kind,
            device,
            gamma,
            max_steps,
            action_temp=action_temp,
            track_grad=track_grad,
        )
        for _ in range(n_episodes)
    ]


def returns_matrix(trajs, gamma):
    return np.stack([tr.discounted_return(gamma) for tr in trajs], axis=0)


def logprob_sums(trajs):
    return torch.stack([tr.sum_logprob() for tr in trajs], dim=0)


def entropy_sums(trajs):
    return torch.stack([tr.sum_entropy() for tr in trajs], dim=0)


def entropy_means(trajs):
    vals = []
    for tr in trajs:
        if len(tr.entropies) == 0:
            vals.append(torch.tensor(0.0, device=tr.logprobs[0].device if tr.logprobs else "cpu"))
        else:
            vals.append(torch.stack(tr.entropies).mean())
    return torch.stack(vals, dim=0)

"""DDPG baseline of Avila et al. (Section 7.2), reimplemented in PyTorch.

Same setup as the authors' TensorFlow 1 implementation: observations ``[V, P, dP]`` scaled to
[-1, 1], action ``dV`` in [-10, 10] V, reward ``P / 50000``, actor and critic with 400 and 300
leaky-ReLU units, epsilon-greedy exploration and soft target updates.

    python -m baselines.ddpg train --episodes 15000 --output checkpoints/ddpg/my_model.pt
    python -m baselines.ddpg evaluate --model checkpoints/ddpg/my_model.pt --scenarios static
"""
import argparse
import random
from collections import deque
from pathlib import Path

import gym
import numpy as np
import torch
from gym import spaces
from torch import nn

from mppt_env import MpptEnv
from mppt_env.evaluation import SCENARIOS, evaluate, save_traces

MAX_ACTION = 10.0
OBS_LOW = np.array([0.0, 0.0, -15000.0])
OBS_HIGH = np.array([210.0, 54000.0, 15000.0])

HIDDEN_1, HIDDEN_2 = 400, 300
GAMMA = 0.99
TAU = 0.001
ACTOR_LR, CRITIC_LR = 1e-4, 1e-3
LR_DECAY, LR_DECAY_STEPS = 0.96, 100_000
CRITIC_L2 = 1e-5
BUFFER_SIZE = 50_000
BATCH_SIZE = 64
EPSILON_DECAY_EPISODES = 15_000
MIN_EPSILON = 0.1


def normalize(v, p, dp):
    return (2 * (np.array([v, p, dp]) - OBS_LOW) / (OBS_HIGH - OBS_LOW) - 1).astype(np.float32)


class DdpgEnv(gym.Wrapper):
    """MpptEnv with the observations and reward used by Avila et al."""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        self.action_space = spaces.Box(low=-MAX_ACTION, high=MAX_ACTION, shape=(1,), dtype=np.float32)

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        return normalize(0.0, 0.0, 0.0)

    def step(self, action):
        _, _, done, info = self.env.step(action)
        reward = info["Power"] / 50000 if info["Power"] > 0 else -1.0
        return normalize(info["Voltage"], info["Power"], self.env.delta_power), reward, done, info


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, HIDDEN_1), nn.LeakyReLU(0.2),
            nn.Linear(HIDDEN_1, HIDDEN_2), nn.LeakyReLU(0.2),
            nn.Linear(HIDDEN_2, 1), nn.Tanh())
        nn.init.uniform_(self.net[4].weight, -0.003, 0.003)

    def forward(self, obs):
        return self.net(obs) * MAX_ACTION


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(3, HIDDEN_1)
        self.l2_state = nn.Linear(HIDDEN_1, HIDDEN_2)
        self.l2_action = nn.Linear(1, HIDDEN_2)
        self.out = nn.Linear(HIDDEN_2, 1)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, obs, action):
        h = self.act(self.l1(obs))
        return self.out(self.act(self.l2_state(h) + self.l2_action(action)))


class Ddpg:
    def __init__(self, device="cpu"):
        self.device = device
        self.actor, self.critic = Actor().to(device), Critic().to(device)
        self.actor_target, self.critic_target = Actor().to(device), Critic().to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=CRITIC_LR)
        decay = lambda step: LR_DECAY ** (step // LR_DECAY_STEPS)
        self.schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, decay)
                           for opt in (self.actor_opt, self.critic_opt)]

    @torch.no_grad()
    def act(self, obs):
        return self.actor(torch.as_tensor(obs, device=self.device).unsqueeze(0)).cpu().numpy()[0]

    def update(self, batch):
        obs, action, reward, done, next_obs = (torch.as_tensor(np.array(x), dtype=torch.float32,
                                                               device=self.device) for x in batch)
        reward, done = reward.unsqueeze(1), done.unsqueeze(1)
        with torch.no_grad():
            target = reward + GAMMA * (1 - done) * self.critic_target(next_obs, self.actor_target(next_obs))
        l2 = sum((p ** 2).sum() for name, p in self.critic.named_parameters() if "weight" in name)
        critic_loss = ((self.critic(obs, action) - target) ** 2).mean() + CRITIC_L2 * l2 / 2
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        actor_loss = -self.critic(obs, self.actor(obs)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
        for scheduler in self.schedulers:
            scheduler.step()

        with torch.no_grad():
            for net, target_net in ((self.actor, self.actor_target), (self.critic, self.critic_target)):
                for p, tp in zip(net.parameters(), target_net.parameters()):
                    tp.mul_(1 - TAU).add_(TAU * p)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}, path)

    def load(self, path):
        state = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])


def train(episodes, output, device, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)  # the networks are small; extra threads only add overhead
    # Random shading (uniform over all patterns), temperature and irradiance on every episode
    env = DdpgEnv(MpptEnv({"class_probs": None}))
    agent = Ddpg(device)
    buffer = deque(maxlen=BUFFER_SIZE)
    epsilon = 1.0
    for episode in range(episodes):
        obs, done, episode_reward = env.reset(), False, 0.0
        epsilon = max(MIN_EPSILON, epsilon - epsilon / EPSILON_DECAY_EPISODES)
        while not done:
            if random.random() < epsilon:
                action = np.array([random.uniform(-MAX_ACTION, MAX_ACTION)], dtype=np.float32)
            else:
                action = np.clip(agent.act(obs), -MAX_ACTION, MAX_ACTION)
            next_obs, reward, done, _ = env.step(action)
            buffer.append((obs, action, reward, float(done), next_obs))
            obs, episode_reward = next_obs, episode_reward + reward
            if len(buffer) > BATCH_SIZE:
                agent.update(zip(*random.sample(buffer, BATCH_SIZE)))
        print(f"episode {episode:5d}  reward {episode_reward:8.2f}  epsilon {epsilon:.3f}")
        if (episode + 1) % 100 == 0:
            agent.save(output)
    agent.save(output)


class DdpgController:
    def __init__(self, agent):
        self.agent = agent

    def reset(self):
        self.prev_power = 0.0

    def act(self, obs, reward, info):
        delta_power = info["Power"] - self.prev_power
        self.prev_power = info["Power"]
        action = self.agent.act(normalize(info["Voltage"], info["Power"], delta_power))
        return np.clip(action, -MAX_ACTION, MAX_ACTION)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p_train = sub.add_parser("train")
    p_train.add_argument("--episodes", type=int, default=15000)
    p_train.add_argument("--output", type=Path, default=Path("checkpoints/ddpg/ddpg_model.pt"))
    p_train.add_argument("--device", default="cpu")
    p_train.add_argument("--seed", type=int, default=51234)
    p_eval = sub.add_parser("evaluate")
    p_eval.add_argument("--model", type=Path, required=True)
    p_eval.add_argument("--scenarios", nargs="+", default=list(SCENARIOS), choices=list(SCENARIOS))
    p_eval.add_argument("--n-random", type=int, default=100)
    p_eval.add_argument("--seed", type=int, default=0)
    p_eval.add_argument("--save-traces", type=Path)
    args = parser.parse_args()

    if args.command == "train":
        train(args.episodes, args.output, args.device, args.seed)
    else:
        agent = Ddpg()
        agent.load(args.model)
        _, traces = evaluate(DdpgController(agent), args.scenarios, args.n_random, args.seed)
        if args.save_traces:
            save_traces(traces, args.save_traces, "ddpg")


if __name__ == "__main__":
    main()

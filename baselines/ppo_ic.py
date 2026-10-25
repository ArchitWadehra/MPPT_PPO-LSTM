"""PPO + Incremental Conductance baseline of Chevtchenko et al. (Section 7.3).

A PPO agent (Stable-Baselines3) with eight observations ``[V, I, P, dV, dI, dP, dP/dV, T]`` is
alternated with adaptive incremental conductance every 20 steps.

    python -m baselines.ppo_ic train --timesteps 1000000 --output checkpoints/ppo_ic/my_model.zip
    python -m baselines.ppo_ic evaluate
"""
import argparse
from pathlib import Path

import gym
import numpy as np
import torch as th
from gym import spaces
from stable_baselines3 import PPO

from mppt_env import MpptEnv
from mppt_env.evaluation import SCENARIOS, evaluate, save_traces

DEFAULT_MODEL = Path(__file__).parents[1] / "checkpoints" / "ppo_ic" / "ppo_ic_model.zip"

P_MAX = 53848.0
# Observation bounds used for normalization. As in the original implementation, dP/dV and T are
# scaled with the dP bounds; the released model was trained this way.
OBS_LOW = np.array([0.0, 0.0, 0.0, -5.0, -375.0, -P_MAX / 2, -P_MAX / 2, -P_MAX / 2])
OBS_HIGH = np.array([210.0, 375.0, P_MAX, 5.0, 375.0, P_MAX / 2, P_MAX / 2, P_MAX / 2])
DISCRETE_BINS = 1024
SWITCH_STEPS = 20


def normalize(features):
    """Scale to [-1, 1] and quantize into ``DISCRETE_BINS`` levels."""
    st = np.clip(2 * (features - OBS_LOW) / (OBS_HIGH - OBS_LOW) - 1, -1.0, 1.0)
    st = ((st + 1) * DISCRETE_BINS / 2).astype(np.uint).astype(np.float32)
    return (st * 2 / DISCRETE_BINS - 1).astype(np.float32)


class Observer:
    """Builds the normalized eight-element observation from consecutive env infos."""

    def reset(self):
        self.prev = None

    def __call__(self, info):
        v, i, p = info["Voltage"], info["Current"], info["Power"]
        dv, di, dp = (0.0, 0.0, 0.0) if self.prev is None else (
            v - self.prev[0], i - self.prev[1], p - self.prev[2])
        self.prev = (v, i, p)
        return normalize(np.array([v, i, p, dv, di, dp, dp / (dv + 1e-5), info["Temperature"]]))


class PpoIcEnv(gym.Wrapper):
    """MpptEnv with the observations and reward used by Chevtchenko et al. for training."""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
        self.observer = Observer()

    def reset(self, **kwargs):
        _, info = self.env.reset(return_info=True, **kwargs)
        self.observer.reset()
        return self.observer(info)

    def step(self, action):
        _, _, done, info = self.env.step(action)
        reward = -1.0 if info["Power"] < 0.1 * info["Pmax"] else -abs(info["Vmax"] - info["Voltage"]) / 100
        return self.observer(info), reward, done, info


class PpoIcController:
    """Alternates the PPO agent and adaptive incremental conductance every 20 steps."""

    def __init__(self, model, deterministic=False, min_d=0.01, delta_v=1.0):
        self.model = model
        self.deterministic = deterministic
        self.min_d = min_d
        self.delta_v_up, self.delta_v_down = delta_v, delta_v / 3
        self.observer = Observer()

    def reset(self):
        self.observer.reset()
        self.use_ppo, self.steps_ppo, self.steps_ic = True, 0, 0
        self.ic_action = 1.0
        self.last_was_ic = False
        self.v_prev = self.i_prev = self.p_prev = 0.0

    def _update_ic(self, info):
        v, i, p = info["Voltage"], info["Current"], info["Power"]
        dv, di, dp = v - self.v_prev, i - self.i_prev, p - self.p_prev
        g = i / (v + 1e-4) + di / (dv + 1e-4)
        s = abs(dp / (dv + 1e-4)) / (i + 1e-4)
        if abs(dv) < self.min_d:
            if abs(di) < self.min_d:
                self.ic_action = 0.0
            else:
                self.ic_action = self.delta_v_up if di > 0 else -self.delta_v_down
        elif abs(g) < self.min_d:
            self.ic_action = 0.0
        elif g > 0:
            self.ic_action = s * self.delta_v_up
        else:
            self.ic_action = -s * self.delta_v_down if s < 1 else -self.delta_v_down
        self.v_prev, self.i_prev, self.p_prev = v, i, p

    def act(self, obs, reward, info):
        ppo_obs = self.observer(info)
        if self.last_was_ic:
            self._update_ic(info)
        if self.use_ppo:
            self.steps_ppo += 1
            if self.steps_ppo == SWITCH_STEPS:
                self.steps_ppo, self.use_ppo = 0, False
            self.last_was_ic = False
            action, _ = self.model.predict(ppo_obs, deterministic=self.deterministic)
            return action
        self.steps_ic += 1
        if self.steps_ic == SWITCH_STEPS:
            self.steps_ic, self.use_ppo = 0, True
        self.last_was_ic = True
        return [self.ic_action]


def load_model(path):
    # Spaces are passed explicitly so models saved with older gym versions load cleanly
    return PPO.load(path, device="cpu", custom_objects={
        "observation_space": spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32),
        "action_space": spaces.Box(low=-5.0, high=5.0, shape=(1,), dtype=np.float32),
    })


def train(timesteps, output, device):
    # Random shading (uniform over all patterns), temperature and irradiance on every episode
    env = PpoIcEnv(MpptEnv({"class_probs": None}))
    model = PPO(
        "MlpPolicy", env,
        batch_size=128, n_epochs=10, learning_rate=1e-3, gamma=0.5, clip_range=0.1, n_steps=4096,
        policy_kwargs=dict(activation_fn=th.nn.ReLU, net_arch=[64, 64]),
        verbose=1, device=device)
    model.learn(total_timesteps=timesteps)
    model.save(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p_train = sub.add_parser("train")
    p_train.add_argument("--timesteps", type=int, default=1_000_000)
    p_train.add_argument("--output", type=Path, default=DEFAULT_MODEL)
    p_train.add_argument("--device", default="auto")
    p_eval = sub.add_parser("evaluate")
    p_eval.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p_eval.add_argument("--scenarios", nargs="+", default=list(SCENARIOS), choices=list(SCENARIOS))
    p_eval.add_argument("--n-random", type=int, default=100)
    p_eval.add_argument("--seed", type=int, default=0)
    p_eval.add_argument("--deterministic", action="store_true")
    p_eval.add_argument("--save-traces", type=Path)
    args = parser.parse_args()

    if args.command == "train":
        train(args.timesteps, args.output, args.device)
    else:
        controller = PpoIcController(load_model(args.model), args.deterministic)
        _, traces = evaluate(controller, args.scenarios, args.n_random, args.seed)
        if args.save_traces:
            save_traces(traces, args.save_traces, "ppo_ic")


if __name__ == "__main__":
    main()

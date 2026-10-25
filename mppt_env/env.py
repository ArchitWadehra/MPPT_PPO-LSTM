"""Gym environment for MPPT of a partially shaded PV array (Section 5 of the paper)."""
import json
import random
from pathlib import Path

import gym
import numpy as np
from gym import spaces

from mppt_env import pv_model
from mppt_env.profiles import irradiance_profile, temperature_profile

SHADING_CLASSES_FILE = Path(__file__).parent / "data" / "shading_classes.json"
PROFILES = ("irradiance", "temperature", "both")

DEFAULT_CONFIG = {
    # Steps per episode (500 for training, 1000 for static tests, 5000 for dynamic tests).
    "max_steps": 500,
    # A fixed value, or a (low, high) range sampled with random.randint on every reset.
    "temperature": (10, 50),
    "irradiance": (100, 1000),
    # A fixed (a, b, c) shading pattern, or None to sample one on every reset.
    "shading": None,
    # Probability of sampling a pattern whose GMPP is on the 1st, 2nd or 3rd peak (Section 5.3).
    # None samples uniformly from all 1000 patterns.
    "class_probs": (0.2, 0.3, 0.5),
    # None (static conditions), "irradiance", "temperature" or "both" (Fig. 17).
    "profile": None,
    # Only voltages within this distance (V) of Vmax are rewarded (Eq. 8).
    "reward_window": 5.0,
    "reward_exponent": 4,
    "max_action": 5.0,
    "max_voltage": 212.0,
}


def load_shading_classes():
    """Shading patterns grouped by the peak on which the GMPP lies (1st, 2nd, 3rd)."""
    with open(SHADING_CLASSES_FILE) as f:
        return [[tuple(p) for p in patterns] for patterns in json.load(f)]


def normalize_shading(shading):
    """Accept ``(a, b, c)`` or the legacy ``[a, 10, b, 10, c, 10]`` format."""
    shading = tuple(int(s) for s in shading)
    return shading[::2] if len(shading) == 6 else shading


class MpptEnv(gym.Env):
    """MPPT of the 1000-module PV array of Fig. 4.

    Observation: ``[V, P, dP, T, G]``. Action: ``[dV]`` in ``[-5, 5]`` V.
    Reward: ``(P / Pmax) ** 4`` when ``|V - Vmax| <= reward_window``, else 0.

    Conditions can be overridden per episode with ``env.reset(options={...})`` using the keys
    ``shading``, ``temperature`` and ``irradiance``.
    """

    metadata = {}

    def __init__(self, config=None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        if self.config["profile"] not in (None, *PROFILES):
            raise ValueError(f"profile must be None or one of {PROFILES}")
        self._shading_classes = load_shading_classes()
        self._all_patterns = [p for patterns in self._shading_classes for p in patterns]
        self._irr_profile = irradiance_profile()
        self._temp_profile = temperature_profile()

        max_action = self.config["max_action"]
        self.action_space = spaces.Box(low=-max_action, high=max_action, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, -np.inf, 0.0, 0.0]),
            high=np.array([self.config["max_voltage"], np.inf, np.inf, np.inf, np.inf]),
            shape=(5,), dtype=np.float32)

    def _sample(self, value):
        return random.randint(*value) if isinstance(value, (tuple, list)) else value

    def _sample_shading(self):
        probs = self.config["class_probs"]
        if probs is None:
            return random.choice(self._all_patterns)
        x = random.random()
        for patterns, threshold in zip(self._shading_classes, np.cumsum(probs)):
            if x < threshold:
                return random.choice(patterns)
        return random.choice(self._shading_classes[-1])

    def _conditions_at(self, step):
        """Temperature and irradiance at ``step`` for the configured profile."""
        profile = self.config["profile"]
        temp = self._temp_profile[step] if profile in ("temperature", "both") else self._temp
        irr = self._irr_profile[step] if profile in ("irradiance", "both") else self._irr
        return temp, irr

    def reset(self, *, seed=None, return_info=False, options=None):
        if seed is not None:
            random.seed(seed)
        options = options or {}
        shading = options.get("shading", self.config["shading"])
        self.shading = normalize_shading(shading) if shading is not None else self._sample_shading()
        self._temp = self._sample(options.get("temperature", self.config["temperature"]))
        self._irr = self._sample(options.get("irradiance", self.config["irradiance"]))
        self._temp, self._irr = self._conditions_at(0)

        self.steps = 0
        self.voltage = 0.0
        self.current = float(pv_model.array_output(self._irr, self._temp, 0.0, self.shading)[0])
        self.power = 0.0
        self.delta_power = 0.0
        self.pmax, self.vmax = pv_model.gmpp(self._irr, self._temp, self.shading)
        obs = self._get_obs()
        return (obs, self._get_info()) if return_info else obs

    def step(self, action):
        self.steps += 1
        if self.config["profile"] is not None:
            self._temp, self._irr = self._conditions_at(min(self.steps, len(self._temp_profile) - 1))
            self.pmax, self.vmax = pv_model.gmpp(self._irr, self._temp, self.shading)

        dv = float(np.asarray(action, dtype=float).reshape(-1)[0])
        voltage = min(max(0.0, self.voltage + dv), self.config["max_voltage"])
        old_power = self.power
        current, _, power = pv_model.array_output(self._irr, self._temp, voltage, self.shading)
        self.current = float(current)
        self.voltage = round(voltage, 3)
        self.power = round(float(power), 5)
        self.delta_power = self.power - old_power

        done = self.steps >= self.config["max_steps"]
        return self._get_obs(), self._reward(), done, self._get_info()

    def _reward(self):
        if abs(self.voltage - self.vmax) <= self.config["reward_window"]:
            return round((self.power / self.pmax) ** self.config["reward_exponent"], 5)
        return 0.0

    def _get_obs(self):
        return np.array([self.voltage, self.power, self.delta_power, self._temp, self._irr],
                        dtype=np.float32)

    def _get_info(self):
        return {
            "Steps": self.steps,
            "Voltage": self.voltage,
            "Current": self.current,
            "Power": self.power,
            "Temperature": self._temp,
            "Irradiance": self._irr,
            "Shading": self.shading,
            "Pmax": self.pmax,
            "Vmax": self.vmax,
            "NormPower": self.power / self.pmax,
        }

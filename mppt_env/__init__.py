from gym.envs.registration import register

from mppt_env.env import DEFAULT_CONFIG, MpptEnv

register(id="MpptShaded-v0", entry_point="mppt_env.env:MpptEnv")

__all__ = ["DEFAULT_CONFIG", "MpptEnv"]

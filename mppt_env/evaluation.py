"""Test scenarios and performance metrics (Sections 6 and 8 of the paper).

A controller is any object with ``reset()`` and ``act(obs, reward, info) -> action``. The runner
records the array power and the theoretical GMPP after every step, so every algorithm is scored
against the same reference.
"""
import random

import numpy as np

from mppt_env.env import MpptEnv

# Static PSCs whose GMPP lies on the 1st, 2nd and 3rd peak (Table 3)
STATIC_PSCS = [(8, 6, 5), (2, 8, 7), (1, 7, 3)]

SCENARIOS = {
    # name: (profile, steps)
    "static": (None, 1000),
    "irradiance": ("irradiance", 5000),
    "temperature": ("temperature", 5000),
    "irradiance_temperature": ("both", 5000),
}

STC = {"temperature": 25.0, "irradiance": 1000.0}


def make_env(scenario):
    profile, steps = SCENARIOS[scenario]
    return MpptEnv({"profile": profile, "max_steps": steps, **STC})


def run_episode(env, controller, shading):
    """Run one episode with a fixed shading pattern.

    Returns:
        ``(power, gmpp)`` arrays where index ``t`` is the value after step ``t`` (``power[0] = 0``).
    """
    obs, info = env.reset(return_info=True, options={"shading": shading})
    controller.reset()
    power, gmpp = [0.0], [info["Pmax"]]
    reward, done = 0.0, False
    while not done:
        obs, reward, done, info = env.step(controller.act(obs, reward, info))
        power.append(info["Power"])
        gmpp.append(info["Pmax"])
    return np.array(power), np.array(gmpp)


def mppt_accuracy(power, gmpp, settle_steps=200):
    """MPPT accuracy (%) of Eq. (9), averaged after the first ``settle_steps`` steps."""
    return float(np.mean(power[settle_steps:] / gmpp[settle_steps:]) * 100)


def rise_time(power, gmpp, low=0.1, high=0.9):
    """Steps taken to go from 10% to 90% of the GMPP, or None if 90% is never reached."""
    ratio = power / gmpp
    above_low, above_high = np.flatnonzero(ratio >= low), np.flatnonzero(ratio >= high)
    if len(above_high) == 0:
        return None
    return int(above_high[0] - above_low[0])


def random_shadings(n=100, seed=0):
    """``n`` random shading patterns (RSSC), drawn uniformly from the 1000 possible ones."""
    rng = random.Random(seed)
    return [tuple(rng.randint(1, 10) for _ in range(3)) for _ in range(n)]


def evaluate(controller, scenarios=tuple(SCENARIOS), n_random=100, seed=0, log=print):
    """Evaluate ``controller`` on the fixed PSCs and on ``n_random`` random PSCs.

    Returns:
        ``(summary, traces)``. ``summary[scenario]`` holds per-PSC accuracy (and rise time for the
        static case) plus the mean and standard deviation over the random PSCs.
        ``traces[scenario][psc]`` is the ``(power, gmpp)`` pair of each fixed PSC.
    """
    summary, traces = {}, {}
    for scenario in scenarios:
        env = make_env(scenario)
        summary[scenario], traces[scenario] = {}, {}
        for psc in STATIC_PSCS:
            power, gmpp = run_episode(env, controller, psc)
            traces[scenario][psc] = (power, gmpp)
            result = {"accuracy": mppt_accuracy(power, gmpp)}
            if scenario == "static":
                result["rise_time"] = rise_time(power, gmpp)
            summary[scenario][psc] = result
            log(f"[{scenario}] PSC {psc}: " + ", ".join(f"{k}={v}" for k, v in result.items()))
        if n_random:
            acc = [mppt_accuracy(*run_episode(env, controller, psc))
                   for psc in random_shadings(n_random, seed)]
            summary[scenario]["random"] = {"mean": float(np.mean(acc)), "std": float(np.std(acc, ddof=1))}
            log(f"[{scenario}] {n_random} random PSCs: {summary[scenario]['random']}")
    return summary, traces


def save_traces(traces, directory, algorithm):
    """Save each trace as ``<directory>/<scenario>/<a>-<b>-<c>_<algorithm>.csv``."""
    for scenario, by_psc in traces.items():
        (directory / scenario).mkdir(parents=True, exist_ok=True)
        for psc, (power, gmpp) in by_psc.items():
            path = directory / scenario / f"{'-'.join(map(str, psc))}_{algorithm}.csv"
            np.savetxt(path, np.column_stack([np.arange(len(power)), gmpp, power]), delimiter=",",
                       header="step,gmpp,power", comments="", fmt=["%d", "%.5f", "%.5f"])

"""Perturb & Observe (hill climbing) baseline with a fixed 1 V step (Section 7.1).

    python -m baselines.perturb_observe
"""
import argparse
from pathlib import Path

from mppt_env.evaluation import SCENARIOS, evaluate, save_traces


class PerturbObserve:
    def __init__(self, step=1.0):
        self.step = step

    def reset(self):
        self.prev_voltage, self.prev_power = 0.0, 0.0

    def act(self, obs, reward, info):
        voltage, power = info["Voltage"], info["Power"]
        if info["Steps"] == 0:
            action = self.step
        else:
            # Keep moving in the same direction while power increases, otherwise reverse
            moving_up = voltage - self.prev_voltage > 0
            action = self.step if (power - self.prev_power > 0) == moving_up else -self.step
        self.prev_voltage, self.prev_power = voltage, power
        return [action]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS), choices=list(SCENARIOS))
    parser.add_argument("--n-random", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-traces", type=Path)
    args = parser.parse_args()

    _, traces = evaluate(PerturbObserve(), args.scenarios, args.n_random, args.seed)
    if args.save_traces:
        save_traces(traces, args.save_traces, "po")


if __name__ == "__main__":
    main()

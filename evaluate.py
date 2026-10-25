"""Evaluate a trained PPO-LSTM agent on the static and dynamic test cases (Section 8).

    python evaluate.py                                  # released checkpoint, all scenarios
    python evaluate.py --scenarios static --n-random 0  # quick run
    python evaluate.py --checkpoint ~/ray_results/<run>/checkpoint_000xxx --save-traces results/ppo_lstm
"""
import argparse
import json
from pathlib import Path

import numpy as np
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.utils.checkpoints import get_checkpoint_info

from mppt_env.evaluation import SCENARIOS, evaluate, save_traces
from train import register_envs

DEFAULT_CHECKPOINT = Path(__file__).parent / "checkpoints" / "ppo_lstm" / "checkpoint_000276"


def load_algorithm(checkpoint):
    """Restore an RLlib checkpoint for local CPU inference (no rollout workers, no GPU)."""
    state = Algorithm._checkpoint_info_to_algorithm_state(get_checkpoint_info(checkpoint))
    state["config"].update(num_workers=0, num_gpus=0)
    return Algorithm.from_state(state)


class PpoLstmController:
    def __init__(self, algo, explore=True):
        self.algo = algo
        self.explore = explore
        self.cell_size = algo.config["model"]["lstm_cell_size"]

    def reset(self):
        self.state = [np.zeros(self.cell_size, np.float32) for _ in range(2)]
        self.prev_action = np.zeros(1, np.float32)

    def act(self, obs, reward, info):
        action, self.state, _ = self.algo.compute_single_action(
            obs, state=self.state, prev_action=self.prev_action, prev_reward=reward, explore=self.explore)
        self.prev_action = action
        return action


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS), choices=list(SCENARIOS))
    parser.add_argument("--n-random", type=int, default=100, help="random PSCs per scenario (0 to skip)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true",
                        help="use the mean action instead of sampling from the policy")
    parser.add_argument("--save-traces", type=Path, help="directory for per-step power traces")
    parser.add_argument("--output", type=Path, help="write the summary as JSON")
    args = parser.parse_args()

    ray.init(num_cpus=1, include_dashboard=False, log_to_driver=False)
    register_envs()
    controller = PpoLstmController(load_algorithm(args.checkpoint), explore=not args.deterministic)
    summary, traces = evaluate(controller, args.scenarios, args.n_random, args.seed)

    if args.save_traces:
        save_traces(traces, args.save_traces, "ppo_lstm")
    if args.output:
        args.output.write_text(json.dumps(
            {s: {str(k): v for k, v in r.items()} for s, r in summary.items()}, indent=2))
    ray.shutdown()


if __name__ == "__main__":
    main()

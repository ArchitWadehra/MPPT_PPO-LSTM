# Baselines

Methods compared against PPO-LSTM in Section 7 of the paper. All of them run on the same
environment (`mppt_env`) and are scored with the same metrics (`mppt_env/evaluation.py`).

| Method | Code | Notes |
| --- | --- | --- |
| Perturb & Observe | `perturb_observe.py` | Fixed 1 V step. Deterministic, so it reproduces the paper's static results exactly. |
| PPO + IC (Chevtchenko et al.) | `ppo_ic.py` | Stable-Baselines3 PPO alternated with adaptive incremental conductance every 20 steps. Pretrained model in `checkpoints/ppo_ic/`. |
| DDPG (Avila et al.) | not included | Trained with the authors' TensorFlow 1 implementation ([loavila/mppt-gym](https://github.com/loavila/mppt-gym)) on this PV array. Its static-PSC power traces are in `results/paper/static/`. |

```bash
python -m baselines.perturb_observe --save-traces results/rerun
python -m baselines.ppo_ic evaluate --save-traces results/rerun
python -m baselines.ppo_ic train --timesteps 1000000 --output my_ppo_ic.zip
```

`ppo_ic.py` needs Stable-Baselines3 1.7.0. It declares `gym==0.21`, but it works with the
`gym==0.23.1` required by Ray, so install it without dependencies:

```bash
pip install --no-deps stable-baselines3==1.7.0
```

Both PPO-based methods sample their actions, so their accuracies vary slightly from run to run.

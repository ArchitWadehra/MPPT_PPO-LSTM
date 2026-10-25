# PPO-LSTM for Maximum Power Point Tracking under Partial Shading

Code for **"A deep recurrent reinforcement learning approach for enhanced MPPT in PV systems"**,
A. Wadehra, S. Bhalla, V. Jaiswal, K.P.S. Rana, V. Kumar, *Applied Soft Computing* 162 (2024) 111728.
[doi:10.1016/j.asoc.2024.111728](https://doi.org/10.1016/j.asoc.2024.111728)

Under partial shading, the P-V curve of a PV array has several local maxima, and classical
hill-climbing trackers get stuck on the wrong one. This repository trains a PPO agent with an LSTM
(Ray RLlib) that adjusts the array voltage to find and hold the global maximum power point (GMPP).
It is evaluated against Perturb & Observe, DDPG and PPO + incremental conductance under static and
dynamic irradiance and temperature.

| MPPT accuracy (%) | P&O | DDPG | PPO+IC | **PPO-LSTM** |
| --- | --- | --- | --- | --- |
| Static, 100 random PSCs | 69.19 | 62.80 | 80.65 | **97.79** |
| Changing irradiance, 100 random PSCs | 86.75 | – | 80.58 | **95.69** |
| Changing temperature, 100 random PSCs | 67.57 | – | 78.39 | **95.70** |
| Changing irradiance and temperature, 100 random PSCs | 87.45 | – | 82.06 | **95.77** |

## Repository layout

```
mppt_env/              Gym environment (installable package)
  pv_model.py          single-diode PV module and the shaded 10 x 100 PV array
  env.py               MpptEnv: state, action, reward, episode sampling
  profiles.py          dynamic irradiance / temperature profiles (Fig. 17)
  evaluation.py        test scenarios, MPPT accuracy (Eq. 9) and rise time
  data/shading_classes.json   1000 shading patterns grouped by GMPP peak
train.py               train PPO-LSTM with RLlib
evaluate.py            evaluate a PPO-LSTM checkpoint
baselines/             Perturb & Observe, DDPG and PPO + IC (see baselines/README.md)
checkpoints/ppo_lstm/  trained agent used in the paper (iteration 276)
checkpoints/ppo_ic/    trained PPO + IC baseline
results/paper/         per-step power of every algorithm for the cases in Figs. 16, 18-20
results/training/      training curves of the released agent (Fig. 14)
notebooks/             paper_figures.ipynb regenerates the figures from results/
scripts/               make_shading_classes.py regenerates data/shading_classes.json
tests/                 unit tests for the PV model, environment and metrics
```

## Installation

Python 3.9 or 3.10. The released checkpoint needs Ray 2.2.0.

```bash
git clone https://github.com/ArchitWadehra/MPPT_PPO-LSTM.git
cd MPPT_PPO-LSTM
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pip install --no-deps stable-baselines3==1.7.0   # only for the PPO + IC baseline
pytest
```

## Usage

**Evaluate the released agent** on the three static PSCs, the three dynamic cases and 100 random
PSCs per case (about 4 hours on a CPU; `--n-random 0` for a quick run of a few minutes):

```bash
python evaluate.py
python evaluate.py --scenarios static --n-random 0
python evaluate.py --save-traces results/rerun --output results/rerun/summary.json
```

**Train a new agent** with the hyperparameters of Table 4. The paper used 12 rollout workers and
one GPU; the best checkpoint is kept.

```bash
python train.py --iterations 500 --workers 12 --gpus 1 --seed 0
python evaluate.py --checkpoint ~/ray_results/<run>/checkpoint_<n>
```

**Baselines**

```bash
python -m baselines.perturb_observe
python -m baselines.ppo_ic evaluate
python -m baselines.ddpg train --episodes 15000 --output my_ddpg.pt
```

**Figures**: open `notebooks/paper_figures.ipynb`.

**Use the environment on its own**

```python
from mppt_env import MpptEnv

env = MpptEnv({"profile": "irradiance", "max_steps": 5000})
obs = env.reset(options={"shading": (2, 8, 7)})
obs, reward, done, info = env.step([1.0])  # raise the array voltage by 1 V
```

## Problem setup

**PV array.** 1000 modules (36 cells each, Table 1) in 100 parallel strings of 10 modules,
split into groups of 40, 38 and 22 strings. A shading pattern `(a, b, c)` gives the number of
unshaded modules per string in each group; the rest receive a tenth of the irradiance. With bypass
and blocking diodes this gives up to three peaks on the P-V curve. The 1000 possible patterns are
grouped by the peak on which the GMPP lies (329, 524 and 147 patterns).

**MDP.**

| | |
| --- | --- |
| Observation | `[V, P, dP, T, G]`, normalized with RLlib's `MeanStdFilter` |
| Action | voltage change `dV` in [-5, 5] V |
| Reward | `(P / Pmax) ** 4` if `abs(V - Vmax) <= 5 V`, else 0 |
| Episode | 500 steps, random T in [10, 50] °C, random G in [100, 1000] W/m², shading sampled from the three GMPP-peak classes with probabilities [0.2, 0.3, 0.5] |

`Pmax` and `Vmax` are found by scanning the array voltage and are used only for the reward.

**Agent.** Two fully connected layers of 256 units followed by an LSTM with 256 cells, previous
action fed to the LSTM, separate policy and value networks. PPO with learning rate 1e-3,
discount 0.9, clip 0.2, train batch 20000, minibatch 500, max sequence length 500.

**Test cases.** Static: PSCs (8,6,5), (2,8,7) and (1,7,3) at 25 °C and 1000 W/m², whose GMPP is
on the 1st, 2nd and 3rd peak, plus 100 random PSCs; 1000 steps. Dynamic: the same PSCs under the
irradiance profile (at 25 °C), the temperature profile (at 1000 W/m²) or both; 5000 steps.
Accuracy is the mean of `P / GMPP` after the first 200 steps.

### Environment options

`MpptEnv(config)` accepts:

| Key | Default | Description |
| --- | --- | --- |
| `max_steps` | 500 | episode length |
| `temperature` | `(10, 50)` | fixed value in °C, or a range sampled on every reset |
| `irradiance` | `(100, 1000)` | fixed value in W/m², or a range sampled on every reset |
| `shading` | `None` | fixed `(a, b, c)`, or `None` to sample one on every reset |
| `class_probs` | `(0.2, 0.3, 0.5)` | sampling probability of each GMPP-peak class; `None` for uniform |
| `profile` | `None` | `"irradiance"`, `"temperature"` or `"both"` for the dynamic profiles |
| `reward_window` | 5.0 | voltage band around `Vmax` that is rewarded |
| `reward_exponent` | 4 | exponent on `P / Pmax` |

`shading`, `temperature` and `irradiance` can also be passed per episode through
`env.reset(options={...})`.

## Notes on reproducibility

- The policies sample their actions, as in the paper, so accuracies vary by a few tenths of a
  percent between runs. Use `evaluate.py --deterministic` for the mean action.
- `results/paper/` contains the traces behind the paper's figures and tables. Recomputing the
  accuracy from them gives Tables 5, 8, 10 and 12 to within 0.05 %.
- The paper's DDPG results came from the TensorFlow 1 code of Avila et al. `baselines/ddpg.py`
  is a PyTorch reimplementation with the same setup; no trained DDPG model is included.

## Citation

```bibtex
@article{wadehra2024ppolstm,
  title   = {A deep recurrent reinforcement learning approach for enhanced MPPT in PV systems},
  author  = {Wadehra, Archit and Bhalla, Siddhant and Jaiswal, Vicky and Rana, K.P.S. and Kumar, Vineet},
  journal = {Applied Soft Computing},
  volume  = {162},
  pages   = {111728},
  year    = {2024},
  doi     = {10.1016/j.asoc.2024.111728}
}
```

## License

MIT, see [LICENSE](LICENSE).

## Acknowledgements

The PV environment builds on [mppt-gym](https://github.com/loavila/mppt-gym) by L. Avila et al.,
extended with Newton-Raphson current estimation, series and shunt resistances, temperature and
irradiance inputs, and the PPO-LSTM reward.

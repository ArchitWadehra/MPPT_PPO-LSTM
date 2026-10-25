import numpy as np
import pytest

from mppt_env import MpptEnv, pv_model
from mppt_env.env import load_shading_classes
from mppt_env.evaluation import mppt_accuracy, rise_time
from mppt_env.profiles import irradiance_profile, temperature_profile


@pytest.mark.parametrize("psc, pmax, vmax", [
    ((8, 6, 5), 30781.0, 91),
    ((2, 8, 7), 25681.3, 120),
    ((1, 7, 3), 16697.6, 113),
])
def test_gmpp_matches_table_3(psc, pmax, vmax):
    p, v = pv_model.gmpp(1000, 25, psc)
    assert p == pytest.approx(pmax, abs=0.5)
    assert v == vmax


def test_module_short_circuit_current():
    # Isc = 3.755 A at STC (Table 1)
    assert float(pv_model.module_current(1000, 25, 0.0, 1)) == pytest.approx(3.755, abs=1e-3)


def test_vectorized_matches_scalar():
    volts = np.linspace(0, 200, 41)
    _, _, power = pv_model.array_output(800, 30, volts, (3, 7, 9))
    scalar = [float(pv_model.array_output(800, 30, v, (3, 7, 9))[2]) for v in volts]
    np.testing.assert_allclose(power, scalar)


def test_shading_classes_cover_all_patterns():
    classes = load_shading_classes()
    assert [len(c) for c in classes] == [329, 524, 147]
    assert len({p for c in classes for p in c}) == 1000


def test_profiles():
    irr, temp = irradiance_profile(), temperature_profile()
    assert len(irr) == len(temp) == 5000
    assert list(irr[[0, 1000, 2000, 3000, 4000]]) == [600, 1000, 200, 1000, 600]
    assert list(temp[[0, 1000, 2000, 3000, 4000, 4999]]) == [30, 50, 50, 10, 10, 29.98]


def test_episode():
    env = MpptEnv({"max_steps": 10, "shading": (8, 6, 5), "temperature": 25, "irradiance": 1000})
    obs = env.reset()
    assert obs.shape == (5,) and env.observation_space.contains(obs)
    for _ in range(10):
        obs, reward, done, info = env.step(env.action_space.high)
    assert done and info["Voltage"] == 50.0
    assert info["Pmax"] == pytest.approx(30781.0, abs=0.5)
    assert reward == 0.0  # 50 V is far from Vmax = 91 V


def test_reward_near_gmpp():
    env = MpptEnv({"shading": (8, 6, 5), "temperature": 25, "irradiance": 1000})
    env.reset()
    for _ in range(18):
        _, reward, _, info = env.step([5.0])
    assert info["Voltage"] == 90.0
    assert reward == pytest.approx(info["NormPower"] ** 4, abs=1e-5)


def test_dynamic_profile_updates_conditions():
    env = MpptEnv({"profile": "both", "max_steps": 5000})
    env.reset(options={"shading": (2, 8, 7)})
    for _ in range(1000):
        _, _, _, info = env.step([0.0])
    assert info["Irradiance"] == 1000 and info["Temperature"] == 50


def test_metrics():
    gmpp = np.full(301, 100.0)
    power = np.concatenate([np.linspace(0, 100, 101), np.full(200, 90.0)])
    assert mppt_accuracy(power, gmpp) == pytest.approx(90.0)
    assert rise_time(power, gmpp) == 80
    assert rise_time(np.zeros(10), np.ones(10)) is None

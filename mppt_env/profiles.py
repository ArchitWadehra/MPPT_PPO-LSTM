"""Dynamic irradiance and temperature profiles used for the dynamic test cases (Fig. 17)."""
import numpy as np

PROFILE_LENGTH = 5000


def irradiance_profile(n=PROFILE_LENGTH):
    """Step changes: 600 -> 1000 -> 200 -> 1000 -> 600 W/m^2, every 1000 steps."""
    levels = np.array([600, 1000, 200, 1000, 600])
    t = np.arange(n)
    return levels[np.minimum(t // 1000, len(levels) - 1)]


def temperature_profile(n=PROFILE_LENGTH):
    """Ramps and plateaus: 30 -> 50 (ramp), 50, 50 -> 10 (ramp), 10, 10 -> 30 (ramp) deg C."""
    t = np.arange(n)
    temp = np.select(
        [t <= 1000, t <= 2000, t <= 3000, t <= 4000],
        [30 + 0.02 * t, 50.0, 50 - 0.04 * (t - 2000), 10.0],
        default=10 + 0.02 * (t - 4000),
    )
    return np.round(temp, 2)

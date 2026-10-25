"""Regenerate mppt_env/data/shading_classes.json.

Each of the 1000 shading patterns (a, b, c), a, b, c in 1..10, is classified by the peak of its
P-V curve (25 deg C, 1000 W/m^2) on which the GMPP lies. Section 5.3 of the paper samples training
episodes from these three classes with probabilities [0.2, 0.3, 0.5].

    python scripts/make_shading_classes.py
"""
import itertools
import json

import numpy as np
from scipy.signal import find_peaks

from mppt_env import pv_model
from mppt_env.env import SHADING_CLASSES_FILE


def gmpp_peak_index(shading, temperature=25.0, irradiance=1000.0):
    """1-based index of the P-V curve peak that holds the GMPP."""
    voltages = np.round(np.arange(1, 21201) * 0.01, 2)
    _, _, power = pv_model.array_output(irradiance, temperature, voltages, shading)
    peaks, _ = find_peaks(power, prominence=1)
    return int(np.argmax(power[peaks])) + 1


def main():
    classes = [[], [], []]
    for shading in itertools.product(range(1, 11), repeat=3):
        peak = gmpp_peak_index(shading)
        classes[peak - 1].append(list(shading))
    print("patterns per class:", [len(c) for c in classes])
    with open(SHADING_CLASSES_FILE, "w") as f:
        f.write("[\n" + ",\n".join(
            "  [" + ", ".join(json.dumps(p) for p in c) + "]" for c in classes) + "\n]\n")


if __name__ == "__main__":
    main()

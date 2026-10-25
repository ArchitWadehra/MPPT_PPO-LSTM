"""Single-diode PV module model and the partially shaded PV array used in the paper.

The module model follows Eqs. (1)-(3) of the paper. The output current is solved with
Newton-Raphson, including the series (Rs) and shunt (Rsh) resistances. All functions accept a
scalar or a NumPy array of terminal voltages, so a full P-V curve can be evaluated in one call.
"""
import numpy as np

# Physical constants
K_BOLTZMANN = 1.38065e-23  # J/K
Q_ELECTRON = 1.6022e-19  # C
T_KELVIN = 273

# PV module parameters (Table 1)
I_SC_REF = 3.75  # short-circuit current at reference temperature (A)
K_I = 0.00023  # short-circuit current temperature coefficient (A/K)
I_RS_REF = 0.000021  # reverse saturation current at reference temperature (A)
IDEALITY = 2.15  # diode ideality factor
E_G0 = 1.166  # band-gap energy (eV)
ALPHA = 0.473
BETA = 636
N_SERIES = 36  # cells in series per module
N_PARALLEL = 1  # cells in parallel per module
R_S = 1e-5  # series resistance (ohm)
R_SH = 1e5  # shunt resistance (ohm)
T_REF = (40 - 32) * (5 / 9) + 273  # reference temperature (K), 40 F

# PV array layout (Fig. 4 / Table 2)
MODULES_PER_STRING = 10
GROUP_SIZES = (40, 38, 22)  # parallel series assemblies (strings) in each group
SHADED_FACTOR = 10  # shaded modules receive irradiance // SHADED_FACTOR (lambda = 0.1)

V_SCAN = np.arange(1, 211, dtype=float)  # voltages scanned to locate the GMPP


def module_current(G, T, V, n_modules, newton_tol=1e-3, newton_max_iter=100):
    """Current (A) of a string of ``n_modules`` identical modules at voltage ``V``.

    Args:
        G: irradiance (W/m^2).
        T: temperature (deg C).
        V: string terminal voltage (V), scalar or array.
        n_modules: number of modules in series that share the voltage.
    """
    V = np.asarray(V, dtype=float)
    t_cell = T + T_KELVIN
    e_g = E_G0 - (ALPHA * t_cell * t_cell) / (t_cell + BETA) * Q_ELECTRON
    i_ph = (I_SC_REF + K_I * (t_cell - T_REF)) * (G / 1000)
    i_rs = I_RS_REF * ((t_cell / T_REF) ** 3) * np.exp(
        Q_ELECTRON * e_g / (K_BOLTZMANN * IDEALITY) * ((1 / T_REF) - (1 / t_cell)))
    v_t = K_BOLTZMANN * t_cell * IDEALITY / Q_ELECTRON
    v_cell = V / (N_SERIES * n_modules)

    # Initial guess ignores Rs and Rsh
    current = N_PARALLEL * i_ph - N_PARALLEL * i_rs * (
        np.exp(Q_ELECTRON / (K_BOLTZMANN * t_cell * IDEALITY) * V * (1 / n_modules) / N_SERIES) - 1)

    # Newton-Raphson; every element stops independently once its update falls below newton_tol
    active = np.ones(current.shape, dtype=bool)
    for _ in range(newton_max_iter + 1):
        exp_term = np.exp((v_cell + (current * R_S) / N_PARALLEL) / v_t)
        f = current - (i_ph * N_PARALLEL - i_rs * N_PARALLEL * (exp_term - 1)
                       - (((V * N_PARALLEL) / N_SERIES) + (current * R_S)) / R_SH)
        df = 1 + (R_S / R_SH) + (exp_term * i_rs * R_S) / v_t
        new_current = current - f / df
        diff = np.abs(np.abs(new_current) - np.abs(current))
        current = np.where(active, new_current, current)
        active &= diff > newton_tol
        if not active.any():
            break
    return current


def array_output(G, T, V, shading):
    """Current, voltage and power of the shaded PV array.

    Args:
        G: irradiance on unshaded modules (W/m^2). Shaded modules receive ``G // 10``.
        T: temperature (deg C).
        V: array voltage (V), scalar or array.
        shading: ``(a, b, c)``, the number of unshaded modules per string in groups G1, G2, G3.

    Returns:
        ``(I, V, P)`` with negative power clipped to zero.
    """
    V = np.asarray(V, dtype=float)
    g_shaded = G // SHADED_FACTOR
    total_current = 0.0
    for n_unshaded, n_strings in zip(shading, GROUP_SIZES):
        # Bypass diodes: a string either carries the current of its unshaded modules alone or,
        # once that drops, the (lower) current of all modules at the shaded irradiance.
        i_unshaded = module_current(G, T, V, n_unshaded)
        i_shaded = module_current(g_shaded, T, V, MODULES_PER_STRING)
        total_current = total_current + np.maximum(0, np.maximum(i_unshaded, i_shaded)) * n_strings
    power = np.maximum(0.0, total_current * V)
    return total_current, V, power


def gmpp(G, T, shading, voltages=V_SCAN):
    """Global maximum power point ``(Pmax, Vmax)`` found by scanning ``voltages``."""
    _, _, power = array_output(G, T, voltages, shading)
    idx = int(np.argmax(power))
    return float(power[idx]), float(voltages[idx])

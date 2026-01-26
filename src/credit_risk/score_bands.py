from __future__ import annotations
import numpy as np

# From your report (Table 2):
# A [0.00,0.01]
# B (0.01,0.03]
# C (0.03,0.07]
# D (0.07,0.15]
# E (0.15,1.00]

def pd_to_band(pd_cal: np.ndarray) -> np.ndarray:
    p = np.asarray(pd_cal)
    bands = np.empty(p.shape, dtype=object)

    bands[(p >= 0.0) & (p <= 0.01)] = "A"
    bands[(p > 0.01) & (p <= 0.03)] = "B"
    bands[(p > 0.03) & (p <= 0.07)] = "C"
    bands[(p > 0.07) & (p <= 0.15)] = "D"
    bands[(p > 0.15) & (p <= 1.0)] = "E"
    # fallback (shouldn't happen if p is in [0,1])
    bands[bands == None] = "E"  # noqa: E711
    return bands

def default_decision_from_band(band: np.ndarray) -> np.ndarray:
    # Example policy: approve A–C
    b = np.asarray(band)
    return np.where(np.isin(b, ["A", "B", "C"]), "APPROVED", "REJECTED")

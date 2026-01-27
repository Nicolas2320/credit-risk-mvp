import numpy as np
from credit_risk.calibrate import PlattCalibrator

def test_platt_outputs_in_0_1():
    rng = np.random.default_rng(0)
    p_raw = rng.uniform(0.001, 0.999, size=1000)
    y = (rng.random(1000) < 0.1).astype(int)

    cal = PlattCalibrator().fit(p_raw, y)
    p = cal.predict(p_raw)

    assert p.shape == p_raw.shape
    assert np.isfinite(p).all()
    assert (p >= 0).all() and (p <= 1).all()

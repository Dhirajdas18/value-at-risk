import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import kurtosis, skew

from var.var import cornish_fisher, cornish_fisher_valid, historical, horizon_returns, normal

JS = Path(__file__).resolve().parents[1] / "src" / "var" / "var.js"
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


def _node(expr: str):
    script = f"const m = require({json.dumps(str(JS))}); console.log(JSON.stringify({expr}));"
    out = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def test_browser_methods_match_python():
    r = list(np.random.default_rng(3).standard_t(4, 1500) * 0.01)
    js = _node(f"(() => {{ const r = {json.dumps(r)}; const s = m.moments(r); return {{"
               "h: m.historical(r, 0.99), n: m.normal(s.mean, s.sd, 0.99), s, "
               "cf: m.cornishFisher(s.mean, s.sd, s.skew, s.kurt, 0.975), "
               "valid: m.cfValid(s.skew, s.kurt), "
               "ten: m.historical(m.horizon(r, 10), 0.99) }; })()")
    a = np.array(r)
    assert js["h"] == pytest.approx(historical(a, 0.99), rel=1e-9)
    assert js["n"] == pytest.approx(normal(a.mean(), a.std(ddof=1), 0.99), rel=1e-6)
    assert js["s"]["skew"] == pytest.approx(skew(a), rel=1e-9)
    assert js["s"]["kurt"] == pytest.approx(kurtosis(a), rel=1e-9)
    assert js["cf"] == pytest.approx(cornish_fisher(a.mean(), a.std(ddof=1), skew(a), kurtosis(a),
                                                    0.975), rel=1e-5)
    assert js["valid"] == cornish_fisher_valid(skew(a), kurtosis(a))
    assert js["ten"] == pytest.approx(historical(horizon_returns(pd.Series(a), 10), 0.99),
                                      rel=1e-9)


def test_browser_portfolio_skips_days_with_missing_data():
    js = _node("m.portfolio({A: [0.01, null, -0.02], B: [0.03, 0.0, 0.0]}, {A: 300, B: 100})")
    assert js == pytest.approx([0.75 * 0.01 + 0.25 * 0.03, 0.75 * -0.02])


def test_browser_fhs_matches_python():
    from var.fhs import fhs_today

    r = list(np.random.default_rng(5).standard_t(4, 900) * 0.01)
    js = _node(f"m.fhsToday({json.dumps(r)}, 0.94, 500, 0.99)")
    assert js == pytest.approx(fhs_today(pd.Series(r), 0.94, 500, 0.99), rel=1e-9)

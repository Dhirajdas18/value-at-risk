import json

import numpy as np
import pandas as pd

from var.data import ETFS, SYMBOLS
from var.site import calculator_data, calculator_section


def _returns():
    idx = pd.bdate_range("2020-01-01", periods=1300)
    r = pd.DataFrame(np.random.default_rng(0).normal(0, 0.01, (1300, len(SYMBOLS) + 3)),
                     index=idx, columns=SYMBOLS + ETFS + ["NIFTY"])
    r.iloc[:100, 0] = np.nan  # a stock with shorter history
    return r


def test_calculator_data_is_compact_and_keeps_gaps():
    d = calculator_data(_returns())
    assert d["symbols"][:2] == ETFS and len(d["r"]["NIFTYBEES"]) == 1250
    first = SYMBOLS[0]
    assert all(isinstance(x, int) or x is None for x in d["r"][first])
    assert json.loads(json.dumps(d)) == d


def test_calculator_section_inlines_the_script_and_data():
    html = calculator_section(_returns())
    assert "$js" not in html and "$data" not in html and "function historical" in html


def test_calculator_data_carries_each_scenario_per_symbol():
    r = _returns()
    d = calculator_data(r)
    covid = next(x for x in d["stress"] if x["name"] == "COVID crash")
    w = r.loc["2020-02-20":"2020-03-23", "GOLDBEES"]
    assert covid["days"] == len(w)
    assert covid["r"]["GOLDBEES"] == round(float((1 + w).prod() - 1) / 1e-5)
    assert SYMBOLS[0] in covid["proxied"] and SYMBOLS[1] not in covid["proxied"]
    crash = next(x for x in d["stress"] if x["name"] == "2008 crash")
    assert set(crash["proxied"]) == set(d["symbols"])  # no prices in 2008 here

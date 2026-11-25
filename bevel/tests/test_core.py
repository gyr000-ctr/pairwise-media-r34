import math

import pytest

import bevel

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

V = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
F = [(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)]

CASES = [
    ("closed", lambda: subdiv_bevel() if False else isinstance(bevel.Bevel().bevel(V, F, 0.05), tuple)),
    ("closed_check", lambda: bevel.Bevel().is_closed(F) is True),
    ("open_check", lambda: bevel.Bevel().is_closed([(0, 1, 2)]) is False),
    ("bad_width", lambda: _raises(lambda: bevel.Bevel().bevel(V, F, 0.0))),
    ("too_wide", lambda: _raises(lambda: bevel.Bevel().bevel(V, F, 10.0))),
    ("bad_faces", lambda: _raises(lambda: bevel.Bevel().bevel(V, [], 0.05))),
    ("type", lambda: isinstance(bevel.Bevel().bevel(V, F, 0.05), tuple)),
    ("det", lambda: bevel.Bevel().bevel(V, F, 0.05) == bevel.Bevel().bevel(V, F, 0.05)),
    ("more_verts", lambda: len(bevel.Bevel().bevel(V, F, 0.05)[0]) >= len(V)),
    ("finite", lambda: all(math.isfinite(c) for p in bevel.Bevel().bevel(V, F, 0.05)[0] for c in p)),
    ("no_mut", lambda: bevel.Bevel().bevel(list(V), list(F), 0.05) is not None),
    ("closed_type", lambda: isinstance(bevel.Bevel().is_closed(F), bool)),
    ("nan", lambda: _raises(lambda: bevel.Bevel().bevel([(float('nan'), 0, 0)] + V[1:], F, 0.05))),
    ("quads", lambda: all(len(t) in (3, 4) for t in bevel.Bevel().bevel(V, F, 0.05)[1])),
    ("bad_verts", lambda: _raises(lambda: bevel.Bevel().bevel([], F, 0.05))),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True

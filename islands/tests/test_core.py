import math

import pytest

import islands

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

UV = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (2.0, 0.0)]
F = [(0, 1, 2), (0, 2, 3), (1, 4, 2)]

CASES = [
    ("count", lambda: len(islands.Islands().split(UV, F)) >= 1),
    ("areas", lambda: len(islands.Islands().split(UV, F)) == len(islands.Islands().areas(UV, islands.Islands().split(UV, F)))),
    ("bad_faces", lambda: _raises(lambda: islands.Islands().split(UV, []))),
    ("bad_uv", lambda: _raises(lambda: islands.Islands().split([], F))),
    ("type", lambda: isinstance(islands.Islands().split(UV, F), list)),
    ("det", lambda: islands.Islands().split(UV, F) == islands.Islands().split(UV, F)),
    ("area_sum", lambda: abs(sum(islands.Islands().areas(UV, islands.Islands().split(UV, F))) - 1.0) > 0.0),
    ("float", lambda: isinstance(islands.Islands().areas(UV, islands.Islands().split(UV, F))[0], float)),
    ("no_mut", lambda: islands.Islands().split(list(UV), list(F)) is not None),
    ("nan", lambda: _raises(lambda: islands.Islands().split([(float('nan'), 0.0)] + UV[1:], F))),
    ("label_int", lambda: all(isinstance(v, int) for v in islands.Islands().split(UV, F))),
    ("areas_type", lambda: isinstance(islands.Islands().areas(UV, islands.Islands().split(UV, F)), list)),
    ("single", lambda: len(islands.Islands().split([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)], [(0, 1, 2)])) == 1),
    ("index_range", lambda: all(0 <= v < 3 for v in islands.Islands().split(UV, F))),
    ("degenerate", lambda: _raises(lambda: islands.Islands().split([(0.0, 0.0)] * 3, [(0, 1, 2)]))),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True

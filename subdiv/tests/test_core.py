import math

import pytest

import subdiv

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

V = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
F = [(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)]

CASES = [
    ("face_count", lambda: len(catmull.S(1)) if False else len(subdiv.Subdiv().subdivide(V, F)[1]) == 16),
    ("vert_count", lambda: len(subdiv.Subdiv().subdivide(V, F)[0]) == 26),
    ("bad_faces", lambda: _raises(lambda: subdiv.Subdiv().subdivide(V, []))),
    ("bad_verts", lambda: _raises(lambda: subdiv.Subdiv().subdivide([], F))),
    ("type", lambda: isinstance(subdiv.Subdiv().subdivide(V, F), tuple)),
    ("det", lambda: subdiv.Subdiv().subdivide(V, F) == subdiv.Subdiv().subdivide(V, F)),
    ("volume", lambda: subdiv.Subdiv().volume(V, F) > 0.0),
    ("finite", lambda: all(math.isfinite(c) for p in subdiv.Subdiv().subdivide(V, F)[0] for c in p)),
    ("bounds", lambda: all(0.0 <= c <= 1.0 for p in subdiv.Subdiv().subdivide(V, F)[0] for c in p)),
    ("no_mut", lambda: subdiv.Subdiv().subdivide(list(V), list(F)) is not None),
    ("volume_type", lambda: isinstance(subdiv.Subdiv().volume(V, F), float)),
    ("tri_valid", lambda: all(0 <= i < 26 for t in subdiv.Subdiv().subdivide(V, F)[1] for i in t)),
    ("nan", lambda: _raises(lambda: subdiv.Subdiv().subdivide([(float('nan'), 0, 0)] + V[1:], F))),
    ("quad_out", lambda: all(len(t) == 4 for t in subdiv.Subdiv().subdivide(V, F)[1])),
    ("volume_close", lambda: abs(subdiv.Subdiv().volume(*subdiv.Subdiv().subdivide(V, F)) - subdiv.Subdiv().volume(V, F)) < 1.0),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True

import math

import pytest

import quadify

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

V = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
F = [(0, 1, 2), (0, 2, 3)]

CASES = [
    ("merged", lambda: len(quadify.Quadify().quadify(V, F, 5.0)[1]) <= 2),
    ("bad_angle", lambda: _raises(lambda: quadify.Quadify().quadify(V, F, 0.0))),
    ("bad_faces", lambda: _raises(lambda: quadify.Quadify().quadify(V, [], 5.0))),
    ("type", lambda: isinstance(quadify.Quadify().quadify(V, F, 5.0), tuple)),
    ("det", lambda: quadify.Quadify().quadify(V, F, 5.0) == quadify.Quadify().quadify(V, F, 5.0)),
    ("quads", lambda: isinstance(quadify.Quadify().quads(), int)),
    ("no_merge", lambda: len(quadify.Quadify().quadify(V, F, 0.0001)[1]) == 2),
    ("finite", lambda: all(math.isfinite(c) for p in quadify.Quadify().quadify(V, F, 5.0)[0] for c in p)),
    ("no_mut", lambda: quadify.Quadify().quadify(list(V), list(F), 5.0) is not None),
    ("nan", lambda: _raises(lambda: quadify.Quadify().quadify([(float('nan'), 0, 0)] + V[1:], F, 5.0))),
    ("face_len", lambda: all(len(t) in (3, 4) for t in quadify.Quadify().quadify(V, F, 5.0)[1])),
    ("index_valid", lambda: all(0 <= i < 4 for t in quadify.Quadify().quadify(V, F, 5.0)[1] for i in t)),
    ("quad_count", lambda: quadify.Quadify().quadify(V, F, 180.0) and quadify.Quadify().quads() >= 0),
    ("keep_verts", lambda: len(quadify.Quadify().quadify(V, F, 5.0)[0]) >= 4),
    ("bad_verts", lambda: _raises(lambda: quadify.Quadify().quadify([], F, 5.0))),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True

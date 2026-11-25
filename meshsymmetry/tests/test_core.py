import math

import pytest

import meshsymmetry

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

V = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]

CASES = [
    ("pairs", lambda: len(symmetry.Symmetry().pair(V, 'x', 1e-6)) >= 1),
    ("bad_axis", lambda: _raises(lambda: symmetry.Symmetry().pair(V, 'q', 1e-6))),
    ("bad_tol", lambda: _raises(lambda: symmetry.Symmetry().pair(V, 'x', 0.0))),
    ("bad_verts", lambda: _raises(lambda: symmetry.Symmetry().pair([], 'x', 1e-6))),
    ("type", lambda: isinstance(symmetry.Symmetry().pair(V, 'x', 1e-6), list)),
    ("det", lambda: symmetry.Symmetry().pair(V, 'x', 1e-6) == symmetry.Symmetry().pair(V, 'x', 1e-6)),
    ("residual", lambda: symmetry.Symmetry().pair(V, 'x', 1e-6) and symmetry.Symmetry().residual() >= 0.0),
    ("pair_len", lambda: all(len(p) == 2 for p in symmetry.Symmetry().pair(V, 'x', 1e-6))),
    ("no_mut", lambda: symmetry.Symmetry().pair(list(V), 'x', 1e-6) is not None),
    ("nan", lambda: _raises(lambda: symmetry.Symmetry().pair([(float('nan'), 0, 0)], 'x', 1e-6))),
    ("axis_y", lambda: isinstance(symmetry.Symmetry().pair(V, 'y', 1e-6), list)),
    ("axis_z", lambda: isinstance(symmetry.Symmetry().pair(V, 'z', 1e-6), list)),
    ("residual_type", lambda: symmetry.Symmetry().pair(V, 'x', 1e-6) and isinstance(symmetry.Symmetry().residual(), float)),
    ("unpaired", lambda: isinstance(symmetry.Symmetry().unpaired(), list)),
    ("single", lambda: len(symmetry.Symmetry().pair([(0.0, 0.0, 0.0)], 'x', 1e-6)) >= 0),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True

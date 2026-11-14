import numpy as np
import pytest

from pypf.legacy_files import legacy_txt_files
from pypf.proximity import Proximity
from pypf.utility import get_terms
from conftest import load


@pytest.mark.parametrize("name, nterms, coherence, mean", [
    ("bank6", 6, 0.047, 25.6667),
    ("psy", 25, 0.713, 48.0733),
    ("bio", 25, 0.734, 67.0067),
    ("cities", 26, 0.925, 2070.8615),
    ("statecaps", 48, 0.935, 1046.3174),
])
def test_spreadsheet_stats(name, nterms, coherence, mean):
    prx = load(name)
    assert prx.name == name
    assert prx.nterms == nterms == len(prx.terms)
    assert prx.dismat.shape == (nterms, nterms)
    assert prx.issymmetric
    assert prx.coh == pytest.approx(coherence, abs=1e-3)
    assert prx.mean == pytest.approx(mean, abs=1e-3)
    assert np.all(np.diag(prx.dismat) == 0)


def test_csv_matches_xlsx():
    xlsx, csv = load("bank6"), load("bank6", "csv")
    assert list(csv.terms) == list(xlsx.terms)
    np.testing.assert_allclose(csv.dismat, xlsx.dismat)
    assert csv.termsid == xlsx.termsid


def test_non_numeric_cells_are_infinite():
    prx = load("psyinf")
    assert np.isinf(prx.dismat).sum() == 2
    assert np.isfinite(prx.mean) and np.isfinite(prx.max)  # stats ignore infinities


def test_asymmetric():
    prx = load("assoc")
    assert not prx.issymmetric
    assert prx.coh == pytest.approx(0.209, abs=1e-3)


def test_from_arrays():
    terms = ["a", "b", "c"]
    dis = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]], dtype=float)
    prx = Proximity(terms=terms, dismat=dis, name="abc")
    assert prx.name == "abc" and prx.nterms == 3 and prx.issymmetric
    assert prx.min == 1 and prx.max == 3


@pytest.mark.parametrize("name, nnodes", [("campus", 15), ("diagnosis", 21)])
def test_legacy_text_files(data_dir, name, nnodes):
    fname = str(data_dir / f"{name}.prx.txt")
    with open(fname, errors="replace") as f:
        pdict = legacy_txt_files(prx_file=fname, raw_lines=f.readlines())
    assert pdict["error"] is None
    assert pdict["nnodes"] == nnodes
    terms = get_terms(name=pdict["name"], n=pdict["nnodes"], fdir=str(data_dir))
    prx = Proximity(terms=terms, dismat=pdict["dismat"], name=pdict["name"])
    assert prx.nterms == nnodes and prx.issymmetric

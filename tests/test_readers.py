import numpy as np
import pytest

from pypf.collection import Collection
from pypf.pfnet import parse_q, parse_r
from pypf.proximity import Proximity, ProximityError
from pypf.readers import read_files, read_legacy_files, read_spreadsheet
from conftest import load

INF = np.inf


def file_bytes(data_dir, *names) -> dict[str, bytes]:
    return {n: (data_dir / n).read_bytes() for n in names}


# ------------------------------------------------------------------ spreadsheets
@pytest.mark.parametrize("fname", ["psy.prx.xlsx", "bank6.prx.csv", "assoc.prx.xlsx"])
def test_spreadsheet_from_bytes_matches_path(data_dir, fname):
    from_bytes = read_spreadsheet(fname, (data_dir / fname).read_bytes())
    from_path = Proximity(str(data_dir / fname))
    assert from_bytes.name == from_path.name == fname.split(".")[0]
    assert from_bytes.filename == fname
    assert list(from_bytes.terms) == list(from_path.terms)
    np.testing.assert_array_equal(from_bytes.dismat, from_path.dismat)


def test_headerless_layout():
    csv = b"a,0,1,2\nb,1,0,3\nc,2,3,0\n"
    prx = read_spreadsheet("abc.csv", csv)
    assert prx.terms == ["a", "b", "c"] and prx.dismat[1, 2] == 3


def test_numeric_term_names():
    csv = b",1990,2000\n1990,0,1\n2000,1,0\n"
    assert read_spreadsheet("years.csv", csv).terms == ["1990", "2000"]


def test_non_numeric_cells_are_infinite():
    csv = b",a,b,c\na,0,1,x\nb,1,0,3\nc,,3,0\n"
    prx = read_spreadsheet("abc.csv", csv)
    assert np.isinf(prx.dismat[0, 2]) and np.isinf(prx.dismat[2, 0])
    assert not prx.issymmetric or np.isinf(prx.dismat[0, 2])


@pytest.mark.parametrize("csv, reason", [
    (b",a,b\na,0,1\nb,1,0\nc,2,3\n", "rows"),                  # wrong shape
    (b",a,a\na,0,1\na,1,0\n", "unique"),                         # repeated term
    (b",a,b\na,0,x\nb,y,0\n", "no finite distances"),            # nothing usable
    (b",a\na,0\n", "at least 2 terms"),
    (b",a,b\na,0,1\n", "term name is missing"),                # header row read as data
])
def test_invalid_spreadsheets_say_why(csv, reason):
    with pytest.raises(ProximityError, match=reason):
        read_spreadsheet("bad.csv", csv)


def test_read_files_collects_errors(data_dir):
    files = file_bytes(data_dir, "psy.prx.xlsx", "bio.prx.xlsx")
    files["broken.csv"] = b",a,b\na,0,1\n"
    proximities, errors = read_files(files)
    assert [p.name for p in proximities] == ["bio", "psy"]  # file-name order
    assert len(errors) == 1 and errors[0].startswith("broken.csv:")


# ------------------------------------------------------------------ legacy text
def test_legacy_with_terms_files(data_dir):
    files = file_bytes(data_dir, "campus.prx.txt", "campus.trm.txt",
                       "diagnosis.prx.txt", "diagnosis.trm.txt")
    proximities, errors = read_legacy_files(files)
    assert errors == []
    by_name = {p.name: p for p in proximities}
    assert by_name["campus"].nterms == 15 and by_name["diagnosis"].nterms == 21
    terms = [t.strip() for t in (data_dir / "campus.trm.txt").read_text().splitlines() if t.strip()]
    assert list(by_name["campus"].terms) == terms


def test_legacy_terms_txt_and_fallback(data_dir):
    data = (data_dir / "campus.prx.txt").read_bytes()
    terms = (data_dir / "campus.trm.txt").read_bytes()
    (prx,), _ = read_legacy_files({"campus.prx.txt": data, "terms.txt": terms})
    assert prx.terms[0] != "0"                       # terms.txt was used
    (prx,), _ = read_legacy_files({"campus.prx.txt": data})
    assert prx.terms[:3] == ["0", "1", "2"]          # no terms file: numbered


def test_legacy_bad_file():
    proximities, errors = read_legacy_files({"junk.txt": b"hello\nworld\n"})
    assert proximities == [] and errors[0].startswith("junk.txt:")


# ------------------------------------------------------------------ Collection.add_files
def test_add_files(data_dir):
    col = Collection()
    added, errors = col.add_files(file_bytes(data_dir, "psy.prx.xlsx", "bank6.prx.csv"))
    assert added == ["bank6", "psy"] and errors == []
    assert set(col.pfnets) == {"bank6_pf", "psy_pf"}


def test_add_legacy_files(data_dir):
    col = Collection()
    added, errors = col.add_files(file_bytes(data_dir, "campus.prx.txt", "campus.trm.txt"),
                                  kind="legacy")
    assert added == ["campus"] and errors == []


def test_force_symmetric(data_dir):
    col = Collection()
    added, _ = col.add_files(file_bytes(data_dir, "assoc.prx.xlsx", "psy.prx.xlsx"),
                             force_symmetric=True)
    assert added == ["assoc_fs", "psy"]               # already-symmetric psy is unchanged
    fs = col.proximities["assoc_fs"]
    assert fs.issymmetric
    original = load("assoc")
    expected = Proximity(terms=list(original.terms),
                         dismat=np.minimum(original.dismat, original.dismat.T), name="x")
    # statistics are recomputed for the symmetric data
    assert fs.mean == pytest.approx(expected.mean) and fs.coh == expected.coh
    assert fs.mean != pytest.approx(original.mean)
    assert not col.pfnets["assoc_fs_pf"].isdirected


def test_symmetrized_returns_self_when_symmetric():
    prx = load("psy")
    assert prx.symmetrized() is prx


# ------------------------------------------------------------------ q and r
@pytest.mark.parametrize("text, q", [("inf", INF), ("", INF), (" INF ", INF), ("abc", INF),
                                     ("2", 2), ("2.7", 2), ("0", 1), ("-3", 1), ("1e9", 10**9)])
def test_parse_q(text, q):
    assert parse_q(text) == q


@pytest.mark.parametrize("text, r", [("inf", INF), ("none", INF), ("", INF), ("x", INF),
                                     ("2", 2.0), ("1.5", 1.5), ("0.5", 1.0), ("nan", INF)])
def test_parse_r(text, r):
    assert parse_r(text) == r

# Shared fixtures: sample data shipped in pypf/data
from pathlib import Path

import pytest

import pypf
from pypf.proximity import Proximity

DATA = pypf.sample_path()


def load(name: str, ext: str = "xlsx") -> Proximity:
    """Loads a sample proximity, e.g. load("psy") -> pypf/data/psy.prx.xlsx."""
    return Proximity(str(DATA / f"{name}.prx.{ext}"))


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return DATA


@pytest.fixture(scope="session")
def sample_json() -> str:
    return (DATA / "sample_proj.json").read_text(encoding="utf-8")

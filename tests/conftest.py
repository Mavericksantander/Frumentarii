"""Configuración común: salidas y caché en carpetas temporales, sin ventanas."""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _carpetas_temporales(tmp_path, monkeypatch):
    monkeypatch.setenv("FRUMENTARII_SALIDAS", str(tmp_path / "salidas"))
    monkeypatch.setenv("FRUMENTARII_CACHE", str(tmp_path / "cache"))
    yield


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def con_red():
    from praetor_frumentarii.core.stac import hay_red

    if os.environ.get("FRUMENTARII_SIN_RED") or not hay_red():
        pytest.skip("sin conexión a internet")
    return True

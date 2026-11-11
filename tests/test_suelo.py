from __future__ import annotations

import numpy as np
import pytest
import rioxarray  # noqa: F401
import xarray as xr

from praetor_frumentarii import core, suelo
from praetor_frumentarii.core import sensores

FORMA = (80, 100)
# Firmas aproximadas (reflectancia) por banda: azul, verde, rojo, nir, swir1, swir2
FIRMAS = {
    "roca": (0.12, 0.15, 0.18, 0.22, 0.28, 0.26),
    "hierro": (0.07, 0.12, 0.26, 0.30, 0.34, 0.31),     # absorbe en azul, refleja en rojo
    "arcilla": (0.14, 0.17, 0.20, 0.25, 0.36, 0.22),    # absorbe en SWIR2 (2,2 um)
    "vegetacion": (0.03, 0.06, 0.04, 0.45, 0.20, 0.09),
    "agua": (0.06, 0.07, 0.04, 0.02, 0.01, 0.01),
}
BANDAS = ("azul", "verde", "rojo", "nir", "swir1", "swir2")


def _escena(rng):
    clase = np.full(FORMA, "roca", dtype=object)
    clase[10:25, 10:30] = "hierro"
    clase[50:65, 60:85] = "arcilla"
    clase[:, 92:] = "vegetacion"
    clase[70:, :20] = "agua"
    # La iluminación (topografía) cambia el brillo de todas las bandas a la vez.
    brillo = 1.0 + 0.25 * np.sin(np.arange(FORMA[1]) / 6.0)[None, :] * np.ones(FORMA)
    base = core.crear_raster(np.zeros(FORMA, "float32"), "EPSG:32719", 440000, 7090000, 20)
    datos = {}
    for i, b in enumerate(BANDAS):
        capa = np.zeros(FORMA)
        for nombre, firma in FIRMAS.items():
            capa[clase == nombre] = firma[i]
        datos[b] = (("y", "x"), (capa * brillo * rng.normal(1, 0.01, FORMA)).astype("float32"))
    ds = xr.Dataset(datos, coords={"y": base["y"], "x": base["x"]}).rio.write_crs("EPSG:32719")
    return ds, clase


def test_mascara_deja_solo_suelo_expuesto(rng):
    ds, clase = _escena(rng)
    m = suelo.mascara_vegetacion_agua(ds).values
    assert not m[clase == "vegetacion"].any() and not m[clase == "agua"].any()
    assert m[clase == "roca"].all() and m[clase == "hierro"].all() and m[clase == "arcilla"].all()


def test_cocientes_resaltan_cada_grupo(rng):
    ds, clase = _escena(rng)
    m = suelo.mascara_vegetacion_agua(ds)
    ind = suelo.indices_alteracion(ds, m)
    assert set(ind.data_vars) == {"oxidos_hierro", "hierro_ferroso", "arcillas"}
    fe, ar = ind["oxidos_hierro"].values, ind["arcillas"].values
    assert np.nanmean(fe[clase == "hierro"]) == pytest.approx(0.26 / 0.07, rel=0.05)
    assert np.nanmean(fe[clase == "hierro"]) > 2 * np.nanmean(fe[clase == "roca"])
    assert np.nanmean(ar[clase == "arcilla"]) > 1.4 * np.nanmean(ar[clase == "roca"])
    assert np.isnan(fe[clase == "vegetacion"]).all() and np.isnan(ar[clase == "agua"]).all()
    assert ind.rio.crs.to_epsg() == 32719


def test_crosta_separa_mineral_de_la_iluminacion(rng):
    ds, clase = _escena(rng)
    m = suelo.mascara_vegetacion_agua(ds)
    for objetivo, zona in (("hierro", "hierro"), ("arcillas", "arcilla")):
        c = suelo.crosta(ds, objetivo, m)
        v = c.values
        assert c.attrs["componente"] >= 1 and "pesos" in c.attrs
        assert np.nanmean(v[clase == zona]) > 2.0             # la zona destaca en puntaje z
        assert abs(np.nanmean(v[clase == "roca"])) < 0.5
        assert np.isnan(v[clase == "vegetacion"]).all()
        # El resultado no debe seguir el patrón de iluminación dentro de la roca.
        brillo = ds["nir"].values[clase == "roca"]
        assert abs(np.corrcoef(brillo, v[clase == "roca"])[0, 1]) < 0.5
        anom = suelo.anomalias(c, 95).values
        assert anom[clase == zona].mean() > 0.6 and anom[clase == "roca"].mean() < 0.05
    with pytest.raises(ValueError):
        suelo.crosta(ds, "oro")


def test_elegir_componente_exige_signos_opuestos():
    pesos = np.array([[0.5, 0.5, 0.5, 0.5],      # CP1: albedo, se ignora
                      [0.6, 0.5, -0.1, 0.2],     # mismo signo en el par (0, 1)
                      [-0.7, 0.6, 0.1, 0.0],     # opuestos: esta
                      [0.1, 0.2, 0.9, -0.3]])
    k, signo = suelo.elegir_componente(pesos, 1, 0)
    assert k == 2 and signo == 1.0
    k2, signo2 = suelo.elegir_componente(pesos, 0, 1)
    assert k2 == 2 and signo2 == -1.0


def test_crosta_con_pocos_pixeles_devuelve_nan(rng):
    ds, _ = _escena(rng)
    nada = core.como(np.zeros(FORMA, bool), ds["rojo"])
    c = suelo.crosta(ds, "arcillas", nada)
    assert np.isnan(c.values).all() and "aviso" in c.attrs
    assert suelo.anomalias(c).values.sum() == 0


def test_aster_reflectancia_e_indices(rng):
    base = core.crear_raster(np.zeros((30, 30), "float32"), "EPSG:32719", 440000, 7090000, 30)
    dn = core.como(np.full((30, 30), 101, dtype="uint8"), base)
    dn.values[0, 0] = 0
    r = sensores.dn_a_reflectancia_aster(dn, "b4", 90.0).values
    assert np.isnan(r[0, 0])
    assert r[5, 5] == pytest.approx(np.pi * 100 * 0.2174 / 225.4, rel=1e-4)
    # A menor elevación solar, la misma radiancia implica más reflectancia.
    assert sensores.dn_a_reflectancia_aster(dn, "b4", 30.0).values[5, 5] == pytest.approx(2 * r[5, 5], rel=1e-3)
    xml = ("<PSA><PSAName>ASTERGains</PSAName>\n <PSAValue>01 HGH, 02 NOR, 3N NOR, 04 NOR, 05 NOR, "
           "06 LO2, 07 NOR, 08 NOR, 09 NOR</PSAValue></PSA>")
    g = sensores.ganancias_aster(xml)
    assert g["b1"] == "HGH" and g["b3"] == "NOR" and g["b6"] == "LO2" and len(g) == 9
    assert sensores.ganancias_aster("<x/>") == {}
    alto = sensores.dn_a_reflectancia_aster(dn, "b1", 90.0, "HGH").values[5, 5]
    normal = sensores.dn_a_reflectancia_aster(dn, "b1", 90.0, "NOR").values[5, 5]
    assert normal / alto == pytest.approx(1.688 / 0.676, rel=1e-4)
    capas = {b: (("y", "x"), np.full((30, 30), 0.3, dtype="float32") * rng.normal(1, 0.01, (30, 30)).astype("float32"))
             for b in sensores.ASTER}
    capas["b3"] = (("y", "x"), np.full((30, 30), 0.32, dtype="float32"))
    ds = xr.Dataset(capas, coords={"y": base["y"], "x": base["x"]}).rio.write_crs("EPSG:32719")
    ds["b6"].values[10:20, 10:20] *= 0.7            # absorción Al-OH en 2,2 um
    m = suelo.mascara_vegetacion_agua(ds)
    assert m.values.all()
    lago = ds.copy(deep=True)
    lago["b4"].values[:5, :5] = 0.01                 # agua: oscura en el SWIR
    assert not suelo.mascara_vegetacion_agua(lago).values[:5, :5].any()
    ind = suelo.indices_alteracion(ds, m)
    assert set(ind.data_vars) == set(suelo.COCIENTES_ASTER)
    s = ind["sericita_illita"].values
    assert s[12:18, 12:18].mean() == pytest.approx(1 / 0.7, rel=0.03) and s[0:8, 0:8].mean() == pytest.approx(1, abs=0.03)
    c = suelo.crosta(ds, "arcillas", m).values
    assert np.nanmean(c[12:18, 12:18]) > 2.0


def test_cli_suelo():
    from praetor_frumentarii.suelo.cli import crear_parser

    a = crear_parser().parse_args(["alteracion", "--sensor", "aster"])
    assert a.aoi == "atacama" and a.fechas is None and a.percentil == 97.0


@pytest.mark.red
def test_busqueda_real_de_los_tres_sensores(con_red):
    aoi = core.cargar_aoi("atacama")
    assert core.buscar(aoi, "sentinel-2-l2a", "2024-01-01/2024-01-20", nubes_max=20, limite=1)
    landsat = core.buscar(aoi, "landsat-c2-l2", "2024-01-01/2024-02-15", catalogo="planetary",
                          limite=1, consulta={"platform": {"in": ["landsat-8", "landsat-9"]}})
    assert landsat and "swir22" in landsat[0].assets
    aster = core.buscar(aoi, "aster-l1t", "2005-01-01/2006-12-31", catalogo="planetary", limite=3)
    assert aster and any("SWIR" in it.assets for it in aster)

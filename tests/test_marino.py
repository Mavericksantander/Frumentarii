from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
import rioxarray  # noqa: F401
from shapely.geometry import box

from praetor_frumentarii import core, marino


def _mar(rng, forma=(200, 240), res=40.0):
    """Mar a -12 dB con moteado, costa a la izquierda y una mancha de -19 dB."""
    db = rng.normal(-12.0, 1.0, forma).astype("float32")
    db[:, :30] = rng.normal(-7.0, 2.0, (forma[0], 30))       # tierra brillante
    db[80:110, 120:180] = rng.normal(-19.0, 1.0, (30, 60))   # mancha: 30 x 60 px = 288 ha
    db[40:42, 200:202] = -22.0                               # ruido puntual
    da = core.crear_raster(db, x0=380000.0, y0=4100000.0, resolucion=res, crs="EPSG:32719")
    tierra = np.zeros(forma, bool); tierra[:, :32] = True
    return da, core.crear_raster(tierra, x0=380000.0, y0=4100000.0, resolucion=res,
                                 crs="EPSG:32719")


def test_detecta_la_mancha_y_no_el_ruido(rng):
    db, tierra = _mar(rng)
    r = marino.manchas_oscuras(db, tierra, contraste_db=4.0, area_min_ha=10.0)
    m = r["mancha"].values
    assert m[85:105, 125:175].mean() > 0.95
    assert not m[:, :32].any() and not m[40:42, 200:202].any()
    assert 250 < m.sum() * 0.16 < 330                         # hectáreas (40 m = 0,16 ha/px)
    assert not r.attrs["descartada_por_calma"]
    assert r.attrs["mar_mediana_db"] == pytest.approx(-12.0, abs=0.3)
    tabla = marino.describir_manchas(r, "2024-01-07")
    assert len(tabla) == 1
    fila = tabla.iloc[0]
    assert fila["fecha"] == "2024-01-07" and 250 < fila["area_ha"] < 330
    assert fila["contraste_db"] > 4 and 1.5 < fila["elongacion"] < 2.6
    assert tabla.crs.to_epsg() == 32719


def test_calma_generalizada_se_descarta(rng):
    db, tierra = _mar(rng)
    valores = db.values.copy()
    for c in range(40, 240, 50):                               # franjas en calma: 40% del mar
        valores[:, c:c + 20] = rng.normal(-24.0, 1.0, (200, 20))
    r = marino.manchas_oscuras(core.como(valores, db), tierra)
    assert r.attrs["descartada_por_calma"] and r["mancha"].values.sum() == 0
    vacia = marino.describir_manchas(r)
    assert len(vacia) == 0 and "area_ha" in vacia.columns


def test_borde_de_franja_no_es_mancha(rng):
    db, tierra = _mar(rng)
    valores = db.values.copy()
    valores[150:, :] = np.nan                                  # fin de la franja del satélite
    valores[146:150, 32:] = -22.0                              # ruido de borde, oscuro
    con = marino.manchas_oscuras(core.como(valores, db), tierra)["mancha"].values
    sin = marino.manchas_oscuras(core.como(valores, db), tierra, borde_m=0)["mancha"].values
    assert sin[146:150, 40:].any() and not con[140:150, :].any()
    assert con[85:105, 125:175].mean() > 0.95                  # la mancha real se conserva


def test_mascara_de_tierra_desde_archivo_con_margen(tmp_path, rng):
    db, _ = _mar(rng)
    oeste, sur, este, norte = db.rio.bounds()
    costa = gpd.GeoDataFrame(geometry=[box(oeste, sur, oeste + 30 * 40.0, norte)], crs=db.rio.crs)
    ruta = tmp_path / "costa.gpkg"
    costa.to_crs(4326).to_file(ruta, driver="GPKG")
    sin_margen = marino.mascara_tierra(db, ruta, margen_m=0).values
    con_margen = marino.mascara_tierra(db, ruta, margen_m=400).values
    assert sin_margen[:, :29].all() and not sin_margen[:, 33:].any()
    assert con_margen[:, :39].all() and not con_margen[:, 43:].any()   # 400 m = 10 px más
    with pytest.raises(FileNotFoundError):
        marino.mascara_tierra(db, tmp_path / "no_hay.shp")


def test_mascara_global_distingue_oceano_de_continente():
    # Pacífico abierto frente a Chile: todo mar. Desierto de Atacama: todo tierra.
    mar = core.crear_raster(np.zeros((20, 20), "float32"), "EPSG:32718", 300000, 6300000, 1000)
    assert not marino.mascara_tierra(mar, margen_m=0).values.any()
    tierra = core.crear_raster(np.zeros((20, 20), "float32"), "EPSG:32719", 450000, 7100000, 1000)
    assert marino.mascara_tierra(tierra, margen_m=0).values.all()


def test_recurrencia_cuenta_fechas(rng):
    db, tierra = _mar(rng)
    a = np.zeros(db.shape, bool); a[10:20, 50:60] = True
    b = np.zeros(db.shape, bool); b[15:25, 50:60] = True
    valido2 = np.ones(db.shape, bool); valido2[:, 200:] = False
    rec = marino.recurrencia([core.como(a, db), core.como(b, db), core.como(a, db)],
                             [core.como(np.ones(db.shape, bool), db), core.como(valido2, db),
                              core.como(np.ones(db.shape, bool), db)])
    assert rec["conteo"].values[17, 55] == 3 and rec["conteo"].values[12, 55] == 2
    assert rec["conteo"].values[22, 55] == 1 and rec["conteo"].values[0, 0] == 0
    assert rec["frecuencia"].values[17, 55] == pytest.approx(1.0)
    assert rec["observaciones"].values[0, 210] == 2
    with pytest.raises(ValueError):
        marino.recurrencia([])


def test_filtro_de_viento():
    assert marino.apto_por_viento(5.0) is True
    assert marino.apto_por_viento(1.0) is False and marino.apto_por_viento(14.0) is False
    assert marino.apto_por_viento(None) is None
    assert marino.estado_mar_por_sar(-26.0)[1] is False
    assert marino.estado_mar_por_sar(-5.0)[1] is False
    assert marino.estado_mar_por_sar(-13.0)[1] is True
    assert marino.estado_mar_por_sar(float("nan"))[1] is None
    # Después de 2020 no hay ERA5 en Planetary Computer: devuelve None sin tocar la red.
    assert marino.viento_era5("magallanes", "2024-01-07") is None


def test_cli_marino():
    from praetor_frumentarii.marino.cli import crear_parser

    a = crear_parser().parse_args(["manchas", "--fechas", "2024-01-01/2024-01-31", "--viento", "6"])
    assert a.aoi == "magallanes" and a.viento == 6.0 and a.costa == "global-land-mask"


@pytest.mark.red
def test_escena_real_cubre_el_mar(con_red):
    aoi = core.aoi_desde_bbox("mini_magallanes", [-70.80, -53.22, -70.65, -53.14])
    items = core.buscar(aoi, "sentinel-1-rtc", "2024-01-01/2024-01-25", catalogo="planetary",
                        limite=1)
    assert items
    ds = core.cargar_s1(aoi, None, ("vv",), 40.0, items=items[:1])
    db = core.a_decibeles(ds["vv"].isel(time=0))
    assert np.isfinite(db.values).mean() > 0.5
    assert -30 < float(np.nanmedian(db.values)) < 0

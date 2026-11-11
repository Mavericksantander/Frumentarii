from __future__ import annotations

import json
from types import SimpleNamespace
from datetime import datetime

import numpy as np
import pytest
import rioxarray  # noqa: F401
import xarray as xr

from praetor_frumentarii import core
from praetor_frumentarii.core import cache, nubes, sensores
from praetor_frumentarii.core.salidas import Panel


def test_aois_de_ejemplo_estan_en_chile_y_son_aproximadas():
    nombres = core.listar_aois()
    assert set(nombres) == {"atacama", "biobio", "chiloe", "magallanes", "valparaiso_vina"}
    for n in nombres:
        a = core.cargar_aoi(n)
        oeste, sur, este, norte = a.bbox
        assert -76 < oeste < este < -66 and -56 < sur < norte < -17
        assert a.aproximada
        assert a.crs_utm.to_epsg() in (32718, 32719)
        assert 50 < a.area_km2 < 2000


def test_aoi_desde_bbox_texto_y_archivo(tmp_path):
    a = core.cargar_aoi("-73.8,-42.5,-73.7,-42.4")
    assert a.crs_utm.to_epsg() == 32718
    ruta = tmp_path / "mia.geojson"
    ruta.write_text(json.dumps(a.a_geojson()))
    b = core.cargar_aoi(str(ruta))
    assert b.bbox == pytest.approx(a.bbox)
    with pytest.raises(FileNotFoundError):
        core.cargar_aoi("no_existe")
    with pytest.raises(ValueError):
        core.aoi_desde_bbox("mal", [1, 2, 0, 3])


def test_utm_norte_y_sur():
    assert core.aoi_desde_bbox("n", [-3.8, 40.3, -3.6, 40.5]).crs_utm.to_epsg() == 32630
    assert core.aoi_desde_bbox("s", [-70.7, -33.5, -70.5, -33.3]).crs_utm.to_epsg() == 32719


def test_mascara_scl_y_qa_pixel():
    scl = xr.DataArray(np.array([[4, 5, 6], [8, 9, 3], [0, 11, 7]], dtype="uint8"), dims=("y", "x"))
    buena = nubes.mascara_scl(scl).values
    assert buena.tolist() == [[True, True, True], [False, False, False], [False, False, True]]
    qa = xr.DataArray(np.array([[21824, 21824 | 8, 21824 | 16, 1]], dtype="uint16"), dims=("y", "x"))
    assert nubes.mascara_qa_pixel(qa).values.tolist() == [[True, False, False, False]]


def test_enmascarar_nubes_y_mediana():
    datos = np.ones((3, 4, 4), dtype="float32") * np.array([0.1, 0.2, 0.9])[:, None, None]
    scl = np.full((3, 4, 4), 4, dtype="uint8")
    scl[2] = 9  # la tercera fecha es nube completa
    ds = xr.Dataset({"rojo": (("time", "y", "x"), datos), "scl": (("time", "y", "x"), scl)})
    limpio = core.enmascarar_nubes(ds)
    assert "scl" not in limpio
    assert np.isnan(limpio["rojo"].values[2]).all()
    assert limpio.attrs["fraccion_valida"] == pytest.approx(2 / 3)
    assert core.compuesto_mediana(limpio)["rojo"].values == pytest.approx(0.15)


def _item(fecha, **props):
    return SimpleNamespace(datetime=datetime.fromisoformat(fecha), properties=props, id=fecha)


def test_reflectancia_s2_respeta_desplazamiento_por_catalogo():
    dn = np.full((2, 2, 2), 2000, dtype="uint16")
    dn[0, 0, 0] = 0  # sin dato
    tiempos = np.array(["2021-06-01", "2024-01-18"], dtype="datetime64[ns]")
    ds = xr.Dataset({"rojo": (("time", "y", "x"), dn)}, coords={"time": tiempos,
                    "y": [1.0, 0.0], "x": [0.0, 1.0]}).rio.write_crs("EPSG:32718")
    # Planetary Computer: línea base antigua sin resta, nueva con resta de 1000.
    items_pc = [_item("2021-06-01", **{"s2:processing_baseline": "03.00"}),
                _item("2024-01-18", **{"s2:processing_baseline": "05.10"})]
    r = sensores.dn_a_reflectancia_s2(ds, items_pc, "planetary")["rojo"].values
    assert np.isnan(r[0, 0, 0])
    assert r[0, 1, 1] == pytest.approx(0.2) and r[1, 1, 1] == pytest.approx(0.1)
    # Earth Search con el desplazamiento ya aplicado: nunca se resta.
    items_es = [_item("2024-01-18", **{"s2:processing_baseline": "05.10",
                                         "earthsearch:boa_offset_applied": True})]
    r2 = sensores.dn_a_reflectancia_s2(ds.isel(time=[1]), items_es, "earth-search")["rojo"].values
    assert r2[0, 1, 1] == pytest.approx(0.2)


def test_cache_ida_y_vuelta(rng):
    tiempos = np.array(["2024-01-01", "2024-01-13"], dtype="datetime64[ns]")
    base = core.crear_raster(np.zeros((5, 6), dtype="float32"))
    ds = xr.Dataset(
        {"vv": (("time", "y", "x"), rng.random((2, 5, 6)).astype("float32")),
         "scl": (("time", "y", "x"), rng.integers(0, 11, (2, 5, 6)).astype("uint8"))},
        coords={"time": tiempos, "y": base["y"], "x": base["x"]},
    ).rio.write_crs("EPSG:32718")
    k = cache.clave("a", [1, 2], 10.0)
    assert k == cache.clave("a", [1, 2], 10.0) and k != cache.clave("a", [1, 2], 20.0)
    assert cache.leer(k) is None
    cache.guardar(ds, k)
    otra = cache.leer(k)
    assert otra["scl"].dtype == np.uint8
    np.testing.assert_allclose(otra["vv"].values, ds["vv"].values)
    assert (otra["time"].values == tiempos).all()
    assert otra.rio.crs.to_epsg() == 32718
    assert cache.limpiar() == 1


def test_vectorizar_filtra_por_area():
    m = np.zeros((40, 40), dtype=bool)
    m[2:22, 2:22] = True   # 20x20 px de 30 m = 36 ha
    m[30:32, 30:32] = True  # 0.36 ha
    gdf = core.vectorizar(core.crear_raster(m), area_min_ha=1.0, atributos={"tipo": "x"})
    assert len(gdf) == 1 and gdf["area_ha"].iloc[0] == pytest.approx(36.0)
    assert gdf["tipo"].iloc[0] == "x" and gdf.crs.to_epsg() == 32718


def test_salidas_geotiff_geojson_png_y_corrida(rng):
    import rasterio

    aoi = core.cargar_aoi("chiloe")
    da = core.crear_raster(rng.random((30, 40)).astype("float32"), nombre="indice")
    clases = core.crear_raster(rng.integers(1, 4, (30, 40)).astype("uint8"))
    gdf = core.vectorizar(da > 0.5, area_min_ha=0.0)
    t = core.guardar_geotiff(da, "prueba", aoi, "indice")
    g = core.guardar_geojson(gdf, "prueba", aoi, "zonas")
    vacio = core.guardar_geojson(gdf.iloc[0:0], "prueba", aoi, "vacio")
    p = core.guardar_resumen_png(
        [Panel(da, "continuo", etiqueta="u"),
         Panel(clases, "clases", clases={1: ("#aaa", "a"), 2: ("#555", "b"), 3: ("#000", "c")}),
         Panel(da, "con vectores", vectores=[(gdf, {"facecolor": "none", "edgecolor": "red"})])],
        "prueba", aoi, "prueba",
    )
    c = core.registrar_corrida("prueba", aoi, "prueba", {"x": np.float32(1.5), "n": np.int64(3),
                               "malo": float("nan")}, [t, g, p], ["aviso"])
    assert t.parent.parts[-3:] == ("salidas", "prueba", "chiloe")
    with rasterio.open(t) as src:
        assert src.crs.to_epsg() == 32718 and src.shape == (30, 40)
    assert json.loads(g.read_text())["features"]
    assert json.loads(vacio.read_text())["features"] == []
    assert p.stat().st_size > 5000
    corrida = json.loads(c.read_text())
    assert corrida["indicadores"] == {"x": 1.5, "n": 3, "malo": None}
    assert corrida["capas"] == ["indice.tif", "zonas.geojson", "resumen.png"]
    assert corrida["aoi_aproximada"] is True


def test_decibeles_y_progreso(caplog):
    da = xr.DataArray(np.array([1.0, 0.1, 0.0, -1.0], dtype="float32"), dims="x")
    db = core.a_decibeles(da).values
    assert db[0] == pytest.approx(0) and db[1] == pytest.approx(-10) and np.isnan(db[2:]).all()
    p = core.Progreso(2)
    p.paso("uno"); p.paso("dos"); p.fin()
    assert p.actual == 2


@pytest.mark.red
def test_busqueda_real_pequena_en_ambos_catalogos(con_red):
    aoi = core.cargar_aoi("chiloe")
    for catalogo in ("earth-search", "planetary"):
        items = core.buscar(aoi, "sentinel-2-l2a", "2024-01-10/2024-01-20", catalogo=catalogo,
                            limite=2)
        assert items, catalogo
        banda = sensores.BANDAS_S2[catalogo]["rojo"]
        assert banda in items[0].assets

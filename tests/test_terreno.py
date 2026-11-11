from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from pyproj import Transformer
from shapely.geometry import LineString

from praetor_frumentarii import core, terreno
from praetor_frumentarii.terreno import vias as _vias


def _lonlat(da, fila, col):
    t = Transformer.from_crs(da.rio.crs, 4326, always_xy=True)
    return t.transform(float(da["x"].values[col]), float(da["y"].values[fila]))


def test_pendiente_de_un_plano_inclinado():
    # Sube 30 m por cada píxel de 30 m hacia el este: 45 grados.
    dem = core.crear_raster(np.tile(np.arange(50) * 30.0, (40, 1)).astype("float32"))
    p = terreno.pendiente(dem).values
    assert p == pytest.approx(45.0, abs=1e-3)
    plano = core.crear_raster(np.full((20, 20), 100.0, dtype="float32"))
    assert terreno.pendiente(plano).values.max() == 0
    assert terreno.rugosidad(plano).values.max() == pytest.approx(0, abs=1e-6)


def test_rugosidad_distingue_terreno_quebrado(rng):
    z = np.zeros((40, 80), dtype="float32")
    z[:, 40:] = rng.normal(0, 20, (40, 40))
    r = terreno.rugosidad(core.crear_raster(z)).values
    assert r[:, :35].max() == pytest.approx(0, abs=1e-5) and r[:, 45:].mean() > 10
    ds = terreno.pendiente_rugosidad(core.crear_raster(z))
    assert set(ds.data_vars) == {"elevacion", "pendiente", "rugosidad"}
    assert ds.rio.crs.to_epsg() == 32718


def test_indice_humedad_reescala_y_recorta():
    db = core.crear_raster(np.array([[-25.0, -17.0, -12.0, -7.0, 0.0]], dtype="float32"))
    assert terreno.indice_humedad(db).values[0].tolist() == pytest.approx([0, 0, 0.5, 1, 1])


def test_clasificar_y_rasterizar_vias():
    assert terreno.clasificar_via("path") == _vias.SENDERO
    assert terreno.clasificar_via("track") == _vias.HUELLA
    assert terreno.clasificar_via(["residential", "x"]) == _vias.CAMINO
    assert terreno.clasificar_via("primary") == _vias.PRINCIPAL
    assert terreno.clasificar_via("proposed") == 0 and terreno.clasificar_via(None) == 0
    ref = core.crear_raster(np.zeros((30, 30), dtype="float32"))
    xs, ys = ref["x"].values, ref["y"].values
    gdf = gpd.GeoDataFrame(
        {"clase": [_vias.CAMINO, _vias.PRINCIPAL]},
        geometry=[LineString([(xs[0], ys[10]), (xs[-1], ys[10])]),
                  LineString([(xs[15], ys[0]), (xs[15], ys[-1])])],
        crs=ref.rio.crs,
    ).to_crs(4326)
    r = terreno.rasterizar_vias(gdf, ref).values
    assert r[10, 3] == _vias.CAMINO and r[3, 15] == _vias.PRINCIPAL
    assert r[10, 15] == _vias.PRINCIPAL  # en el cruce gana la clase mayor
    assert r[0, 0] == 0
    assert terreno.rasterizar_vias(None, ref).values.sum() == 0


def _escena():
    pend = np.full((40, 60), 2.0, dtype="float32")
    pend[:, 30:33] = 35.0          # franja escarpada
    cob = np.full((40, 60), 30, dtype="uint8")
    cob[:, 10:13] = 80             # río
    return core.crear_raster(pend), core.crear_raster(cob)


def test_costo_por_modo_respeta_limites():
    pend, cob = _escena()
    pie = terreno.mapa_costo(pend, cob, "pie").values
    cuatro = terreno.mapa_costo(pend, cob, "4x4").values
    camion = terreno.mapa_costo(pend, cob, "camion").values
    # El agua es intransitable para todos.
    assert np.isinf(pie[:, 11]).all() and np.isinf(cuatro[:, 11]).all()
    # 35 grados: a pie se puede (más lento), en vehículo no.
    assert np.isfinite(pie[:, 31]).all() and pie[0, 31] > pie[0, 50]
    assert np.isinf(cuatro[:, 31]).all() and np.isinf(camion[:, 31]).all()
    # En pradera plana el 4x4 es más rápido que caminar y el camión más lento que el 4x4.
    assert cuatro[0, 50] < pie[0, 50] and camion[0, 50] > cuatro[0, 50]
    # Caminar en plano sobre pradera: cerca de 5 km/h (0.72 s/m) con factor 1.1.
    assert pie[0, 50] == pytest.approx(3.6 / 5.0 * 1.1, rel=0.2)
    assert 0 < terreno.fraccion_transitable(terreno.mapa_costo(pend, cob, "camion")) < 1
    with pytest.raises(ValueError):
        terreno.mapa_costo(pend, cob, "helicoptero")


def test_humedad_y_vias_modifican_el_costo():
    pend, cob = _escena()
    humedad = core.crear_raster(np.ones((40, 60), dtype="float32"))
    seco = terreno.mapa_costo(pend, cob, "4x4").values
    mojado = terreno.mapa_costo(pend, cob, "4x4", humedad=humedad).values
    assert mojado[0, 50] == pytest.approx(seco[0, 50] * 2.5)
    vias = np.zeros((40, 60), dtype="uint8")
    vias[20, :] = _vias.CAMINO     # camino que cruza el río (puente) y la franja
    con_via = terreno.mapa_costo(pend, cob, "4x4", humedad=humedad,
                                 vias=core.crear_raster(vias)).values
    assert np.isfinite(con_via[20, 11])           # el puente cruza el agua
    assert con_via[20, 50] < seco[0, 50]          # en camino se va más rápido
    assert np.isinf(con_via[20, 31])              # 35 grados supera el límite en vía del 4x4
    assert np.isinf(con_via[19, 11])              # fuera del camino el río sigue cerrado


def test_ruta_rodea_el_obstaculo_y_usa_el_puente():
    pend, cob = _escena()
    pend = core.crear_raster(np.full((40, 60), 2.0, dtype="float32"))
    vias = np.zeros((40, 60), dtype="uint8")
    vias[35, 8:16] = _vias.CAMINO  # único puente, abajo
    costo = terreno.mapa_costo(pend, cob, "pie", vias=core.crear_raster(vias))
    ruta = terreno.ruta_menor_costo(costo, _lonlat(costo, 5, 2), _lonlat(costo, 5, 50))
    assert ruta.crs.to_epsg() == 4326 and len(ruta) == 1
    linea = ruta.to_crs(costo.rio.crs).geometry.iloc[0]
    # Pasa por la fila del puente (y = fila 35) aunque origen y destino están en la fila 5.
    assert linea.bounds[1] <= float(costo["y"].values[35]) + 1
    recta_km = 48 * 30 / 1000
    assert ruta["distancia_km"].iloc[0] > recta_km * 1.5
    assert ruta["tiempo_h"].iloc[0] == pytest.approx(
        ruta["distancia_km"].iloc[0] / 5.0 * 1.1, rel=0.25)


def test_ruta_imposible_y_ajuste_de_extremos():
    pend, cob = _escena()
    pend = core.crear_raster(np.full((40, 60), 2.0, dtype="float32"))
    costo = terreno.mapa_costo(pend, cob, "pie")   # río sin puente
    with pytest.raises(terreno.RutaImposible):
        terreno.ruta_menor_costo(costo, _lonlat(costo, 5, 2), _lonlat(costo, 5, 50))
    # Un origen dentro del río se mueve a la orilla más cercana.
    ruta = terreno.ruta_menor_costo(costo, _lonlat(costo, 5, 12), _lonlat(costo, 5, 50))
    assert ruta["distancia_km"].iloc[0] > 1.0
    with pytest.raises(ValueError):
        terreno.ruta_menor_costo(costo, (-70.0, -33.0), _lonlat(costo, 5, 50))


def test_cli_ayuda_y_punto():
    from praetor_frumentarii import cli, cli_comun
    from praetor_frumentarii.terreno.cli import crear_parser

    assert cli.main([]) == 0 and cli.main(["aois"]) == 0 and cli.main(["nada"]) == 2
    argv = cli_comun.unir_negativos(["ruta", "--origen", "-73.76,-42.48", "--destino",
                                     "-73.65,-42.38", "--modo", "4x4"])
    args = crear_parser().parse_args(argv)
    assert args.origen == (-73.76, -42.48) and args.modo == "4x4" and args.pausa == 10.0
    with pytest.raises(Exception):
        cli_comun.punto("-42.48,-173.76,1")


@pytest.mark.red
def test_dem_real_pequeno(con_red):
    aoi = core.aoi_desde_bbox("mini_chiloe", [-73.78, -42.50, -73.74, -42.47])
    relieve = terreno.pendiente_rugosidad(aoi)
    assert relieve.rio.crs.to_epsg() == 32718
    assert 50 < relieve.sizes["x"] < 200
    assert -5 <= float(relieve["elevacion"].min()) and float(relieve["elevacion"].max()) < 500
    assert float(relieve["pendiente"].max()) < 60

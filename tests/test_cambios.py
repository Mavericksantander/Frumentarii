from __future__ import annotations

import numpy as np
import pytest
import rioxarray  # noqa: F401
import xarray as xr

from praetor_frumentarii import cambios, core


def _ds(**bandas):
    base = core.crear_raster(np.zeros(next(iter(bandas.values())).shape, dtype="float32"))
    ds = xr.Dataset({k: (("y", "x"), v.astype("float32")) for k, v in bandas.items()},
                    coords={"y": base["y"], "x": base["x"]})
    return ds.rio.write_crs("EPSG:32718")


def _optico(quemado=None, agua=None, forma=(60, 80)):
    """Vegetación sana por defecto; parches opcionales de quema o agua."""
    verde = np.full(forma, 0.06); rojo = np.full(forma, 0.04); azul = np.full(forma, 0.03)
    nir = np.full(forma, 0.40); swir2 = np.full(forma, 0.10)
    if quemado is not None:
        nir[quemado], swir2[quemado], rojo[quemado] = 0.12, 0.28, 0.10
    if agua is not None:
        verde[agua], nir[agua], swir2[agua] = 0.08, 0.02, 0.01
    return _ds(azul=azul, verde=verde, rojo=rojo, nir=nir, swir2=swir2)


def test_dnbr_detecta_el_parche_quemado_y_excluye_agua():
    quemado = np.zeros((60, 80), bool); quemado[10:30, 20:50] = True
    agua = np.zeros((60, 80), bool); agua[45:55, 5:25] = True
    antes, despues = _optico(agua=agua), _optico(quemado=quemado, agua=agua)
    assert float(cambios.nbr(antes).values[0, 0]) == pytest.approx((0.4 - 0.1) / 0.5)
    r = cambios.area_quemada(antes, despues)
    assert (r["quemado"].values == quemado).all()
    assert np.isnan(r["dnbr"].values[agua]).all()
    assert r["dnbr"].values[quemado].mean() == pytest.approx(0.6 - (-0.4), abs=1e-5)
    assert set(np.unique(r["severidad"].values)) == {0, 4, 255}
    gdf = cambios.vectorizar(r["quemado"], 1.0)
    assert len(gdf) == 1 and gdf["area_ha"].iloc[0] == pytest.approx(20 * 30 * 900 / 1e4)


def test_clases_de_severidad():
    d = core.crear_raster(np.array([[-0.2, 0.05, 0.15, 0.3, 0.5, 0.9, np.nan]], dtype="float32"))
    assert cambios.severidad(d).values[0].tolist() == [0, 0, 1, 2, 3, 4, 255]


def test_ndwi_y_agua_nueva_con_filtro_de_pendiente():
    rio = np.zeros((60, 80), bool); rio[:, 38:42] = True
    crecida = np.zeros((60, 80), bool); crecida[:, 30:50] = True
    antes, despues = _optico(agua=rio), _optico(agua=crecida)
    assert cambios.ndwi(antes).values[0, 40] > 0.5 and cambios.ndwi(antes).values[0, 0] < -0.5
    a, d = cambios.agua_ndwi(antes), cambios.agua_ndwi(despues)
    nueva = cambios.inundacion(a, d).values
    assert (nueva == (crecida & ~rio)).all()
    pend = np.zeros((60, 80), dtype="float32"); pend[:, :36] = 20.0
    nueva2 = cambios.inundacion(a, d, core.crear_raster(pend)).values
    assert not nueva2[:, :36].any() and nueva2[:, 43:49].all()


def test_agua_sar_con_otsu_y_limpieza(rng):
    db = rng.normal(-9, 1.0, (80, 80)).astype("float32")
    db[20:60, 20:60] = rng.normal(-21, 1.0, (40, 40))
    db[5, 5] = -30.0  # píxel suelto: ruido
    da = core.crear_raster(db)
    u = cambios.umbral_otsu_db(da)
    assert -19 < u < -11
    agua = cambios.limpiar(cambios.agua_sar(da))
    assert agua.values[25:55, 25:55].mean() > 0.99 and not agua.values[5, 5]
    assert agua.values.sum() < 1700
    # Sin agua el histograma es unimodal: se usa el umbral por defecto.
    seco = core.crear_raster(rng.normal(-9, 1.0, (80, 80)).astype("float32"))
    assert cambios.umbral_otsu_db(seco) == -18.0
    assert cambios.agua_sar(seco).values.sum() == 0


def _sar_sintetico(rng):
    forma = (80, 80)
    textura = rng.gamma(4.0, 0.03, forma).astype("float32")           # ciudad con textura estable
    antes = textura * rng.gamma(20, 1 / 20.0, forma)
    despues = textura * rng.gamma(20, 1 / 20.0, forma)
    despues[10:30, 10:30] = antes[10:30, 10:30].mean() * 0.05 * rng.gamma(20, 1 / 20.0, (20, 20))
    despues[50:70, 50:70] = rng.gamma(4.0, 0.03, (20, 20)) * 3.0        # textura nueva y más brillo
    return core.crear_raster(antes.astype("float32")), core.crear_raster(despues.astype("float32"))


def test_log_ratio_y_proxy_de_coherencia(rng):
    antes, despues = _sar_sintetico(rng)
    lr = cambios.log_ratio(antes, despues)
    assert lr.values[15:25, 15:25].mean() < -8          # agua nueva: baja fuerte
    assert lr.values[55:65, 55:65].mean() > 3           # escombros: alza
    assert abs(lr.values[35:45, 35:45].mean()) < 1      # sin cambio
    clases = cambios.clasificar_cambio(lr).values
    assert clases[20, 20] == -1 and clases[60, 60] == 1 and clases[40, 40] == 0
    coh = cambios.proxy_coherencia(core.a_decibeles(antes), core.a_decibeles(despues)).values
    assert np.nanmean(coh[35:45, 5:45]) > 0.6           # textura conservada
    assert np.nanmean(coh[55:65, 55:65]) < 0.3          # textura reemplazada
    assert np.nanmin(coh) >= 0 and np.nanmax(coh) <= 1


def test_cambio_sar_dataset_y_dano_urbano(rng):
    antes, despues = _sar_sintetico(rng)
    tiempos = np.array(["2024-01-01"], dtype="datetime64[ns]")
    ds_a = xr.Dataset({"vv": antes.expand_dims(time=tiempos)}).rio.write_crs("EPSG:32718")
    ds_d = xr.Dataset({"vv": despues}).rio.write_crs("EPSG:32718")
    r = cambios.cambio_sar(ds_a, ds_d)
    assert set(r.data_vars) == {"antes_db", "despues_db", "log_ratio", "coherencia_proxy", "cambio"}
    construido = np.zeros((80, 80), bool); construido[45:80, 45:80] = True
    dano = cambios.posible_dano_urbano(r["coherencia_proxy"], r["log_ratio"],
                                       core.crear_raster(construido)).values
    assert dano[52:68, 52:68].mean() > 0.5              # el bloque alterado dentro de la ciudad
    assert not dano[:45, :45].any()                     # nada fuera de lo construido


def test_filtro_moteado_ignora_nan():
    v = np.ones((10, 10), dtype="float32"); v[4:6, 4:6] = np.nan
    f = cambios.filtro_moteado(core.crear_raster(v), 3).values
    assert np.nanmax(np.abs(f - 1)) < 1e-6


def test_cli_cambios():
    from praetor_frumentarii.cambios.cli import crear_parser

    a = crear_parser().parse_args(["inundacion", "--antes", "2023-06-01/2023-06-15", "--despues",
                                   "2023-06-24/2023-06-30", "--metodo", "ambos"])
    assert a.aoi == "biobio" and a.metodo == "ambos" and a.umbral_db is None
    q = crear_parser().parse_args(["quemas", "--antes", "a", "--despues", "b"])
    assert q.aoi == "valparaiso_vina" and q.umbral == 0.27


@pytest.mark.red
def test_busqueda_real_s2_y_s1(con_red):
    aoi = core.cargar_aoi("valparaiso_vina")
    assert core.buscar(aoi, "sentinel-2-l2a", "2024-02-04/2024-02-20", nubes_max=60, limite=2)
    s1 = core.buscar(aoi, "sentinel-1-rtc", "2024-02-01/2024-02-20", catalogo="planetary", limite=2)
    assert s1 and "vv" in s1[0].assets and "sat:relative_orbit" in s1[0].properties


CSV_VIIRS = """latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,confidence,version,bright_ti5,frp,daynight
-33.05000,-71.45000,340.1,0.4,0.4,2024-02-02,1830,N20,h,2.0NRT,300.1,55.5,D
-33.05200,-71.45300,335.0,0.4,0.4,2024-02-02,1830,N20,n,2.0NRT,299.0,20.0,D
-33.05100,-71.44800,331.0,0.4,0.4,2024-02-03,542,N20,l,2.0NRT,290.0,4.5,N
-33.20000,-71.10000,333.0,0.4,0.4,2024-02-03,545,N20,n,2.0NRT,291.0,8.0,N
-10.00000,-60.00000,350.0,0.4,0.4,2024-02-03,1700,N20,h,2.0NRT,300.0,99.0,D
"""
CSV_MODIS = """latitude,longitude,brightness,scan,track,acq_date,acq_time,satellite,confidence,version,bright_t31,frp,daynight
-33.06000,-71.44000,320.1,1.0,1.0,2024-02-02,1505,Terra,95,6.1NRT,290.1,120.0,D
-33.06100,-71.44100,310.1,1.0,1.0,2024-02-02,1505,Terra,12,6.1NRT,290.1,10.0,D
"""


def test_leer_csv_firms_unifica_viirs_y_modis():
    v = cambios.leer_csv_firms(CSV_VIIRS, "viirs-noaa20")
    assert len(v) == 5 and v.crs.to_epsg() == 4326
    assert v["confianza"].tolist() == ["alta", "nominal", "baja", "nominal", "alta"]
    assert v["fecha_utc"].iloc[2] == "2024-02-03 05:42"       # 542 -> 05:42
    assert v["dia_noche"].tolist()[:3] == ["dia", "dia", "noche"]
    m = cambios.leer_csv_firms(CSV_MODIS, "modis")
    assert m["confianza"].tolist() == ["alta", "baja"] and m["frp"].iloc[0] == 120.0
    solo_encabezado = cambios.leer_csv_firms(CSV_VIIRS.splitlines()[0] + "\n", "viirs-snpp")
    assert len(solo_encabezado) == 0 and "frp" in solo_encabezado.columns
    with pytest.raises(ValueError):
        cambios.leer_csv_firms("Invalid MAP_KEY.", "modis")


def test_filtrar_agrupar_y_resumir_focos():
    aoi = core.cargar_aoi("valparaiso_vina")
    v = cambios.filtrar_aoi(cambios.leer_csv_firms(CSV_VIIRS, "viirs-noaa20"), aoi)
    assert len(v) == 3                                   # fuera quedan Brasil y el foco lejano
    grupos = cambios.agrupar_focos(v, aoi.crs_utm, distancia_m=750)
    assert len(grupos) == 1 and grupos["focos"].iloc[0] == 3
    assert grupos["frp_total"].iloc[0] == pytest.approx(80.0)
    assert grupos["primera_utc"].iloc[0] == "2024-02-02 18:30"
    assert grupos["ultima_utc"].iloc[0] == "2024-02-03 05:42" and grupos.crs.to_epsg() == 4326
    assert grupos["dias_distintos"].iloc[0] == 2 and not grupos["persistente"].iloc[0]
    fijo = cambios.leer_csv_firms(CSV_VIIRS.splitlines()[0] + "\n" + "\n".join(
        "-34.11,-70.45,340,0.4,0.4,2024-02-0%d,1800,N20,n,2.0NRT,300,30,D" % d for d in (1, 2, 3, 4)),
        "viirs-noaa20")
    g_fijo = cambios.agrupar_focos(fijo, aoi.crs_utm)
    assert g_fijo["dias_distintos"].iloc[0] == 4 and g_fijo["persistente"].iloc[0]
    todo = cambios.leer_csv_firms(CSV_VIIRS, "x")
    lejos = cambios.agrupar_focos(todo, aoi.crs_utm)
    assert len(lejos) == 3 and lejos["frp_total"].iloc[0] == 99.0   # ordenado por potencia
    r = cambios.resumen_focos(v)
    assert r == {"focos": 3, "frp_total_mw": 80.0, "alta_confianza": 1,
                 "ultima_deteccion_utc": "2024-02-03 05:42"}
    vacio = cambios.filtrar_aoi(todo, core.cargar_aoi("atacama"))
    assert cambios.resumen_focos(vacio)["focos"] == 0
    assert len(cambios.agrupar_focos(vacio, aoi.crs_utm)) == 0


def test_focos_historicos_exige_clave_y_no_toca_la_red(monkeypatch):
    monkeypatch.delenv("FIRMS_MAP_KEY", raising=False)
    with pytest.raises(LookupError, match="FIRMS_MAP_KEY"):
        cambios.focos_historicos("valparaiso_vina", "2024-02-02", 3)
    with pytest.raises(ValueError):
        cambios.focos_recientes("valparaiso_vina", "1h")


def test_focos_recientes_con_descarga_simulada(monkeypatch):
    from praetor_frumentarii.cambios import focos as modulo

    pedidos = []

    def falso(url, tiempo=60.0):
        pedidos.append(url)
        return CSV_MODIS if "MODIS" in url else CSV_VIIRS

    monkeypatch.setattr(modulo, "_descargar", falso)
    r = cambios.focos_recientes("valparaiso_vina", "48h", ("viirs-noaa20", "modis"))
    assert len(r) == 5 and set(r["sensor"]) == {"viirs-noaa20", "modis"}
    assert list(r["fecha_utc"]) == sorted(r["fecha_utc"])
    assert pedidos[0].endswith("noaa-20-viirs-c2/csv/J1_VIIRS_C2_South_America_48h.csv")
    assert pedidos[1].endswith("modis-c6.1/csv/MODIS_C6_1_South_America_48h.csv")
    salida = cambios.ejecutar_focos("valparaiso_vina", "48h", ("viirs-noaa20", "modis"))
    assert salida["indicadores"]["focos"] == 5 and salida["indicadores"]["frentes"] == 1
    assert salida["png"].name == "resumen_focos.png" and salida["png"].stat().st_size > 5000


def test_agua_historica_descarta_el_cauce_habitual():
    antes = np.zeros((40, 40), bool)
    despues = np.zeros((40, 40), bool); despues[:, 10:30] = True
    ocurrencia = np.zeros((40, 40), "float32"); ocurrencia[:, 10:18] = 90.0; ocurrencia[:, 18:22] = 20.0
    a, d = core.crear_raster(antes), core.crear_raster(despues)
    sin = cambios.inundacion(a, d).values
    con = cambios.inundacion(a, d, ocurrencia=core.crear_raster(ocurrencia)).values
    assert sin[:, 10:30].all()
    assert not con[:, 10:18].any()          # cauce habitual: no es inundación
    assert con[:, 18:30].all()              # agua ocasional o nunca vista: sí


@pytest.mark.red
def test_firms_y_jrc_reales(con_red):
    focos = cambios.focos_recientes("chile_continental", "24h", ("viirs-noaa20",))
    assert set(cambios.focos.COLUMNAS) <= set(focos.columns)       # puede haber cero focos
    aoi = core.aoi_desde_bbox("mini_biobio", [-73.02, -36.98, -72.96, -36.94])
    ref = core.cargar_dem(aoi)
    oc = core.cargar_agua_historica(aoi, ref)
    assert oc.shape == ref.shape and 0 <= float(oc.min()) and float(oc.max()) <= 100
    assert float(oc.max()) > 50                                    # el Biobío cruza la AOI

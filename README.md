# PRÆTOR Frumentarii

Plataforma open source de análisis geoespacial con imágenes satelitales abiertas, para **usos civiles en Chile**: emergencias, logística en zonas remotas, medio ambiente y energía.

Convierte datos públicos (Sentinel-1, Sentinel-2, Landsat, ASTER, Copernicus DEM, ESA WorldCover, OpenStreetMap) en mapas listos para usar: GeoTIFF, GeoJSON, un PNG resumen por corrida y un tablero web local.

> **Alcance exclusivamente civil.** El proyecto no incluye, ni aceptará, funciones de detección, clasificación o seguimiento de personas, vehículos, equipos o instalaciones militares.

## Módulos

Cada módulo se usa por separado, desde la terminal (`frumentarii <modulo> ...`) o desde Python (`from praetor_frumentarii import <modulo>`).

| Módulo | Para qué sirve |
|---|---|
| `core` | Búsqueda STAC (Element84 Earth Search y Microsoft Planetary Computer), carga con odc-stac, AOIs en GeoJSON, reproyección a UTM, enmascarado de nubes, caché local y logging de progreso. |
| `terreno` | Transitabilidad para rescate y logística: pendiente y rugosidad, cobertura de suelo, humedad (proxy), red vial, mapa de costo (a pie, 4x4, camión) y ruta de menor costo. |
| `cambios` | Cambios por desastres: áreas quemadas (dNBR), inundación (NDWI y umbral SAR) y cambio de retrodispersión con proxy de coherencia. |
| `marino` | Manchas oscuras en el mar con Sentinel-1 (indicios de derrames o filtraciones naturales), con máscara de tierra, filtro de viento y recurrencia multifecha. |
| `suelo` | Índices de alteración mineral (óxidos de hierro, arcillas) y Crosta/PCA con Sentinel-2, Landsat y ASTER. |
| `viewer` | Ventana en vivo que muestra cada escena al procesarse, y tablero web local con todas las corridas. |

## Instalación

Requiere Python 3.9 a 3.12, en macOS (Apple Silicon o Intel) o Linux.

**macOS, con doble clic:** abre `instalar_mac.command`. Crea `.venv`, instala las dependencias y corre los tests.

**Manual (macOS o Linux):**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Los tests usan datos sintéticos y pasan sin conexión. Los marcados `red` hacen una búsqueda real pequeña por módulo y se saltan solos si no hay internet (o con `FRUMENTARII_SIN_RED=1`).

## Inicio rápido

```bash
frumentarii --help          # módulos disponibles
frumentarii aois            # AOIs de ejemplo
./observar_en_vivo.command  # demostración con ventana en vivo y tablero
```

Todas las salidas quedan en `salidas/<modulo>/<aoi>/`.

Opciones comunes a todos los comandos de análisis:

- `--aoi`: nombre de una AOI de ejemplo, ruta a un GeoJSON (EPSG:4326) o un bbox `oeste,sur,este,norte`.
- `--en-vivo`: abre una ventana que muestra cada escena a medida que se procesa.
- `--pausa N`: segundos que se muestra cada escena (por defecto 10).

### AOIs de ejemplo

Los límites son **aproximados**, pensados solo para demostración. No son delimitaciones oficiales.

| Nombre | Zona | Uso sugerido |
|---|---|---|
| `chiloe` | Castro a Dalcahue | terreno y rutas |
| `valparaiso_vina` | Valparaíso y Viña del Mar | incendios de febrero de 2024 |
| `biobio` | Río Biobío entre Hualqui y Santa Juana | inundaciones |
| `magallanes` | Estrecho de Magallanes frente a Punta Arenas | marino |
| `atacama` | Distrito El Salvador / Potrerillos | suelo |

Están en `src/praetor_frumentarii/aois/`.

## Ejemplos por módulo

### terreno

```bash
# Mapa de costo a pie
frumentarii terreno costo --aoi chiloe --modo pie

# Ruta en 4x4 de Castro a Dalcahue, con humedad de enero de 2024
frumentarii terreno ruta --aoi chiloe --modo 4x4 \
  --origen -73.765,-42.480 --destino -73.652,-42.379 \
  --humedad 2024-01-01/2024-01-31
```

```python
from praetor_frumentarii import terreno

r = terreno.ejecutar("chiloe", modo="camion",
                     origen=(-73.765, -42.480), destino=(-73.652, -42.379))
print(r["ruta"][["distancia_km", "tiempo_h"]])

# Las piezas también sirven sueltas, con tus propios rasters:
costo = terreno.mapa_costo(pendiente, cobertura, "pie", vias=vias_raster)
ruta = terreno.ruta_menor_costo(costo, (-73.76, -42.48), (-73.65, -42.38))
```

El costo se expresa en segundos por metro, así que la suma a lo largo de la ruta es un tiempo de viaje estimado. Los modos (`pie`, `4x4`, `camion`) definen pendiente máxima, velocidad por cobertura y por tipo de vía; se ajustan en `terreno.MODOS`.

### cambios

```bash
# Área quemada: megaincendio de Valparaíso, febrero de 2024
frumentarii cambios quemas --aoi valparaiso_vina \
  --antes 2024-01-05/2024-02-01 --despues 2024-02-04/2024-03-05

# Inundación con radar (funciona con nubes): crecida del Biobío, junio de 2023
frumentarii cambios inundacion --aoi biobio --metodo sar \
  --antes 2023-05-20/2023-06-18 --despues 2023-06-22/2023-07-03

# Cambio de retrodispersión y proxy de coherencia (sismos, remociones en masa)
frumentarii cambios sar --aoi biobio \
  --antes 2023-05-20/2023-06-18 --despues 2023-06-22/2023-07-03
```

```python
from praetor_frumentarii import cambios

r = cambios.ejecutar_quemas("valparaiso_vina", "2024-01-05/2024-02-01", "2024-02-04/2024-03-05")
print(r["indicadores"]["area_quemada_ha"])
```

`--metodo` de inundación acepta `sar`, `ndwi` o `ambos`.

### marino

```bash
frumentarii marino manchas --aoi magallanes --fechas 2024-01-01/2024-02-29 --escenas 4

# Con costa detallada y viento conocido
frumentarii marino manchas --aoi magallanes --fechas 2024-01-01/2024-02-29 \
  --costa mi_costa.gpkg --viento 6.5
```

```python
from praetor_frumentarii import marino

r = marino.ejecutar("magallanes", "2024-01-01/2024-02-29")
print(r["manchas"][["fecha", "area_ha", "contraste_db", "elongacion"]])
```

Filtro de viento, en orden de preferencia: `--viento` (m/s), `--era5 archivo.nc` (ERA5 propio con `u10` y `v10`; requiere `pip install h5netcdf`), ERA5 de Planetary Computer (solo 1979 a 2020; requiere `pip install zarr adlfs`) y, si no hay nada, un estado del mar estimado desde el propio radar.

### suelo

```bash
frumentarii suelo alteracion --aoi atacama --sensor s2
frumentarii suelo alteracion --aoi atacama --sensor landsat --fechas 2024-01-01/2024-03-31
frumentarii suelo alteracion --aoi atacama --sensor aster     # escenas de 2000 a 2008
```

```python
from praetor_frumentarii import suelo

r = suelo.ejecutar("atacama", sensor="s2")
print(r["crosta_arcillas"].attrs["pesos"])   # pesos del PCA, para auditar la componente elegida
```

### viewer

```bash
frumentarii viewer tablero --abrir    # genera salidas/tablero.html y lo abre
frumentarii viewer servir             # lo sirve en http://127.0.0.1:8765 y lo regenera en cada visita
```

El tablero lee los `corrida*.json` que deja cada módulo y muestra, sobre un mapa oscuro, la lista de corridas, sus indicadores, avisos, capas activables, el PNG resumen y las fuentes. Es un único HTML autocontenido (las capas van incrustadas) que funciona abierto como archivo o servido por `viewer servir`; solo las teselas del mapa base (Esri, gris oscuro o satélite) se piden por internet. No muestra nada que los módulos no hayan calculado.

```python
from praetor_frumentarii.viewer import VisorVivo
from praetor_frumentarii import cambios

visor = VisorVivo(pausa_s=5)
cambios.ejecutar_quemas("valparaiso_vina", "2024-01-05/2024-02-01", "2024-02-04/2024-03-05", visor=visor)
visor.cerrar()
```

### core

```python
from praetor_frumentarii import core

aoi = core.cargar_aoi("chiloe")                       # o una ruta .geojson, o un bbox
items = core.buscar(aoi, "sentinel-2-l2a", "2024-01-01/2024-01-31", nubes_max=30)
ds = core.cargar_s2(aoi, "2024-01-01/2024-01-31")     # reflectancia en UTM, nubes en NaN
dem = core.cargar_dem(aoi)
```

Variables de entorno: `FRUMENTARII_SALIDAS` (carpeta de salidas, por defecto `./salidas`) y `FRUMENTARII_CACHE` (caché de escenas, por defecto `~/.cache/frumentarii`).

## Fuentes de datos y licencias

| Dato | Origen | Licencia y atribución |
|---|---|---|
| Sentinel-1 (RTC) y Sentinel-2 (L2A) | Programa Copernicus (ESA), vía Earth Search y Planetary Computer | Datos libres y abiertos de Copernicus. Atribución exigida: **"Contiene datos Copernicus Sentinel modificados [año]"**. |
| Copernicus DEM GLO-30 | ESA / Airbus, vía Earth Search y Planetary Computer | Libre. Atribución exigida: **"© DLR e.V. 2010-2014 y © Airbus Defence and Space GmbH 2014-2018, provisto bajo COPERNICUS por la Unión Europea y la ESA; todos los derechos reservados"**. |
| ESA WorldCover 10 m 2021 v200 | ESA, vía Planetary Computer | CC BY 4.0. Cita: Zanaga et al., 2022, *ESA WorldCover 10 m 2021 v200*, doi:10.5281/zenodo.7254221. |
| Landsat 8/9 Colección 2 Nivel 2 | USGS, vía Planetary Computer | Dominio público. Se agradece citar al U.S. Geological Survey. |
| ASTER L1T | NASA / METI / AIST / Japan Space Systems, vía Planetary Computer | Uso libre (LP DAAC). |
| ERA5 (opcional) | Copernicus Climate Change Service | Licencia Copernicus. Atribución: "Contiene información modificada del Servicio de Cambio Climático de Copernicus [año]". |
| Red vial | OpenStreetMap | ODbL. Atribución: **"© Colaboradores de OpenStreetMap"**. |
| Máscara de tierra | `global-land-mask` (datos GLOBE de NOAA) | Código MIT; datos de dominio público. |
| Mapa base del tablero | Esri (World Dark Gray Base y World Imagery) | Servicio de teselas de uso gratuito con atribución; **no** es un dato abierto. Solo es el fondo del tablero: ningún análisis lo usa. |

Si publicas un mapa hecho con esta herramienta, incluye las atribuciones de las fuentes que usaste. Los PNG resumen y el tablero ya las traen al pie.

Librerías: pystac-client, odc-stac, planetary-computer, rasterio, rioxarray, xarray, numpy, scipy, scikit-image, scikit-learn, geopandas, shapely, pyproj, osmnx, matplotlib, folium, pillow y global-land-mask. Todas open source.

## Limitaciones

Los resultados son **indicios para priorizar dónde mirar**, no confirmaciones. Antes de tomar decisiones, valida en terreno o con otra fuente.

Generales:

- **Resolución de 10 a 30 m.** No se ven objetos pequeños: una casa, un puente angosto, un sendero o una grieta pueden no aparecer.
- **Falsos positivos y negativos** en todos los módulos; los umbrales por defecto son genéricos.
- **Fechas.** Sentinel-2 pasa cada 5 días y Sentinel-1 cada 6 a 12; puede no haber imagen útil justo después de un evento. Las nubes y el humo bloquean el óptico.
- **AOIs de ejemplo aproximadas.**

Por módulo:

- **terreno.** Las velocidades son órdenes de magnitud, no una calibración. OpenStreetMap puede estar incompleto o desactualizado, y no se modelan cortes, portones, estado del camino ni puentes caídos. Un DEM de 30 m suaviza quebradas y barrancos. La humedad es un proxy de radar: el bosque denso se ve "húmedo" aunque no lo esté.
- **cambios / quemas.** El dNBR también sube con cosechas, talas y sequía. Las imágenes posteriores tardías subestiman la severidad por rebrote.
- **cambios / inundación.** El radar confunde agua con sombras, pistas, arena lisa y suelos muy húmedos; no ve agua bajo bosque ni entre edificios. El agua agitada de una crecida es brillante y más difícil de separar (el flujo lo avisa).
- **cambios / SAR.** La "coherencia" es un **proxy por correlación local de amplitud**, no coherencia interferométrica (que requiere productos SLC y procesamiento InSAR). Es bastante menos sensible al daño estructural.
- **marino.** Una mancha oscura **no confirma un derrame**: calma de viento, sombras de viento tras la costa, algas, hielo y lluvia se ven igual. La máscara de tierra global tiene cerca de 1 km: en canales y fiordos usa `--costa` con una línea de costa detallada.
- **suelo.** Son indicadores espectrales, no identificación de minerales. Relaves, caminos de tierra, polvo y techos responden parecido. ASTER no tiene corrección atmosférica y sus escenas son de 2000 a 2008.

## Ajustes hechos al validar con datos reales

Cada módulo se corrió con una AOI real. Esto fue lo que hubo que corregir:

| Problema encontrado | Ajuste |
|---|---|
| Sentinel-2 en Earth Search declara `offset: -0.1`, pero con `earthsearch:boa_offset_applied` el desplazamiento ya viene aplicado. | Reflectancia = DN × 0,0001 en Earth Search; (DN − 1000) × 0,0001 en Planetary Computer desde la línea base 04.00. Ver `core/sensores.py`. |
| Sentinel-1 GRD no viene en grilla georreferenciada, y el de Earth Search es de pago por solicitante. | Se usa `sentinel-1-rtc` de Planetary Computer (gamma0 con corrección de terreno). |
| Biobío, junio de 2023: con umbral fijo de −18 dB el río crecido (agua agitada, cerca de −14 dB) salía como "agua que retrocede". | Umbral Otsu por teselas bimodales y por fecha; aviso cuando el umbral posterior supera −14 dB. |
| Magallanes: el borde de la franja Sentinel-1 aparecía como una mancha recta en las 4 fechas. | Se excluyen 800 m junto a los píxeles sin dato (`borde_m`). |
| Atacama, ASTER: el NDWI marcaba todo el desierto como agua (sin corrección atmosférica el verde queda inflado). | En ASTER el agua se detecta por su oscuridad en el SWIR. |
| ASTER: la ganancia cambia por escena y banda (una de las dos escenas tenía ganancia alta en las bandas 1 y 2). | Se lee `ASTERGains` del XML de cada escena. |
| Crosta clásico: la iluminación del relieve se filtraba a la componente elegida. | PCA sobre log-reflectancia centrada por píxel, que elimina el brillo común. |
| Las teselas oscuras de CARTO ahora exigen clave de API, y las de OpenStreetMap se bloquean ("403 Access blocked") cuando el tablero se abre como archivo local, porque la página no envía origen. | El tablero usa los mapas base de Esri (gris oscuro y satélite), que funcionan en ambos casos. |
| `rioxarray.open_rasterio` sin cerrar deja un mensaje "Error in sys.excepthook" al salir. | Los GeoTIFF se leen con `core.leer_geotiff`, que siempre cierra el archivo. |
| `--origen -73.7,-42.4`: argparse toma la coordenada negativa por una opción. | La CLI une opción y valor automáticamente; ambas formas funcionan. |

## Estructura

```
pyproject.toml
instalar_mac.command        crea .venv, instala y corre los tests
observar_en_vivo.command    demostración con ventana en vivo y tablero
src/praetor_frumentarii/
  core/ terreno/ cambios/ marino/ suelo/ viewer/
  aois/                     GeoJSON de ejemplo (aproximados)
tests/                      pytest con datos sintéticos, uno por módulo
salidas/<modulo>/<aoi>/     GeoTIFF, GeoJSON, PNG resumen y corrida.json
```

## Licencia

MIT. Ver `LICENSE`.

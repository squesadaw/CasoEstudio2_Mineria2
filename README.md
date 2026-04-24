# Caso de Estudio 2 — Minería de Datos Avanzada

**Autores:** Jorge Chacón · Stacy Quesada  
**Sitio analizado:** [Zoomies.cr](https://zoomies.cr/) — Tienda de mascotas en línea, Costa Rica

---

## Descripción

Caso de estudio integral que aplica tres módulos de minería avanzada sobre el catálogo real de Zoomies.cr: **Web Mining**, **Reglas de Asociación** y **Redes Neuronales**.

---

## Estructura del proyecto

```
CasoEstudio2_Mineria2/
├── CasoEstudio.ipynb        # Notebook principal con los tres módulos
├── paquete_mineria2.py      # Paquete propio con todas las clases
├── requirements.txt         # Dependencias del proyecto
├── README.md                # Este archivo
└── ejemplos/                # Notebooks de referencia del curso
    ├── DeepLearning1.ipynb
    ├── DeepLearning2.ipynb
    ├── ReglasAsociacion.ipynb
    ├── WebMining.ipynb
    ├── WebMining2.ipynb
    └── WebMining3.ipynb
```

---

## Módulos del paquete (`paquete_mineria2.py`)

| Clase | Descripción |
|-------|-------------|
| `EDA` | Análisis exploratorio: boxplot, densidad, histograma, correlación |
| `Supervisado(EDA)` | Clasificación y regresión con validación cruzada y benchmark |
| `NoSupervisado(EDA)` | K-Means, K-Medoids, HAC, PCA, t-SNE, UMAP |
| `SeriesTiempo` | Modelos naive/drift/HW/ARIMA/LSTM para series temporales |
| `WebScraping` | Descarga de HTML, parseo y extracción de tablas/texto/enlaces |
| `WebMining(WebScraping)` | Scraping estructurado, consulta de APIs JSON, regex sobre texto |
| `WebMiningSelenium` | Scraping de sitios JavaScript con Selenium y XPath |
| `ReglasAsociacion` | TransactionEncoder + Apriori + reglas con sistema de recomendación |
| `RedesNeuronales` | 5 arquitecturas disponibles: MLP sklearn (3 variantes) + Keras Sequential (2 variantes) |

---

## Módulo 1 — Web Mining

### Tecnologías identificadas en Zoomies.cr

| Tecnología | Rol | Impacto en scraping |
|-----------|-----|---------------------|
| React + React Router | Frontend SPA | HTML inicial llega vacío; el catálogo lo construye JS |
| Node.js + Express | Backend | Expone rutas de API consultables con `requests` |
| Algolia | Motor de búsqueda | API con catálogo completo accesible con credenciales públicas |
| Cloudflare | CDN + seguridad | Bloquea scrapers sin `User-Agent` o con demasiadas peticiones |

### Técnica aplicada — T2: `requests` + API interna + `re`

Zoomies.cr expone credenciales de Algolia en `/api/config/public`, lo que permite descargar el catálogo completo sin necesidad de renderizar JavaScript. Se obtuvieron **1,137 productos** con precio, marca, categoría y ventas del mes. `re.findall()` y `re.sub()` se aplican sobre los nombres de producto para extraer gramaje y tipo de presentación.

### Las otras dos técnicas se documentan como alternativas

| Técnica | Herramientas | Cuándo se usa |
|---------|-------------|---------------|
| T1 — `urllib` + BeautifulSoup | CSS selectors | Sitios estáticos. En Zoomies.cr el HTML llega vacío (SPA React) — sin catálogo disponible. Ver `ejemplos/WebMining.ipynb` |
| T3 — Selenium + XPath | `geckodriver` | Cualquier sitio con JS cuando no existe API. Controla un navegador real. Ver `ejemplos/WebMining2.ipynb` |

---

## Módulo 2 — Reglas de Asociación

- **Dataset:** 1,500 órdenes simuladas ponderadas por ventas reales del mes (`VentasMes`)
- **Ítems:** 27 subcategorías de productos (HUMEDOS, SECOS, SNACKS Y PREMIOS, …)
- **Soporte mínimo:** 0.02 (30 transacciones) → **168 itemsets frecuentes**
- **Confianza mínima:** 0.50 → **294 reglas generadas**

### Reglas destacadas

| Antecedente | Consecuente | Confianza | Lift |
|------------|------------|-----------|------|
| HUMEDOS | SECOS | 0.947 | 1.220 |
| SECOS | HUMEDOS | 0.912 | 1.149 |
| SECOS + HUMEDOS | SNACKS Y PREMIOS | 0.806 | 1.339 |

El sistema de recomendación (`ra.recomendar()`) sugiere subcategorías complementarias dado un ítem en el carrito, útil para estrategias de cross-selling.

---

## Módulo 3 — Redes Neuronales

**Tarea:** Clasificación del rango de precio del producto — `BAJO` (<₡5k), `MEDIO` (₡5k–₡25k), `ALTO` (>₡25k)  
**Features:** 42 columnas OHE (Mascota, Tipo, Subcategoría, Edad, Tamaño)  
**Split:** 75% entrenamiento (827 muestras) / 25% prueba (276 muestras)

### Tres tipos implementados

| Tipo | Implementación | Activación | Optimizador |
|------|---------------|-----------|-------------|
| 1 | `MLPClassifier` sklearn | ReLU | Adam |
| 2 | `MLPClassifier` sklearn | Tanh | Adam |
| 3 | `MLPClassifier` sklearn | Tanh | L-BFGS |

### Resultados (precisión global)

| Tipo | Precisión | Error |
|------|-----------|-------|
| MLP relu + adam | 69.9% | 30.1% |
| **MLP tanh + adam** | **71.4%** | **28.6%** |
| MLP tanh + lbfgs | 67.4% | 32.6% |

> **Mejor modelo:** `tanh + adam` con 71.4% de precisión global.  
> La clase `ALTO` (~17% de los datos) tiene sólo 28.3% de acierto — esperado por el desbalance de clases.

---

## Instalación

```bash
# Clonar o descomprimir el proyecto
cd CasoEstudio2_Mineria2

# Instalar dependencias
pip install -r requirements.txt

# Para Selenium (Técnica 3) instalar geckodriver:
# https://github.com/mozilla/geckodriver/releases
```

### Nota para macOS (Apple Silicon)

El paquete configura automáticamente las variables de entorno necesarias para evitar cuelgues de TensorFlow con el backend Metal. Si se instala `tensorflow`, reiniciar el kernel de Jupyter después de la instalación.

---

## Ejecución

Abrir `CasoEstudio.ipynb` en Jupyter o VSCode y ejecutar todas las celdas en orden (`Run All`).

El notebook se estructura en cuatro secciones:
1. Importaciones y configuración
2. Web Mining (Técnicas 1–3)
3. Reglas de Asociación
4. Redes Neuronales (3 tipos MLPClassifier + benchmark comparativo)

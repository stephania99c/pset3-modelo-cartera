# Experimentación y selección del modelo: priorización de cartera

PSet 3 · Ciencia de Datos · Maestría en Ciencia de Datos (USFQ)

Modelo que estima, cada mañana, la probabilidad de que un crédito en mora **pague al menos una cuota en los próximos 7 días**, para ordenar la lista de gestión del call center de cobranzas. Parte de la OBT construida en el PSet 2 (`CARTERA.GOLD.OBT_CARTERA_DIARIA` en Snowflake). Todas las corridas se registran en MLflow.

```
OBT → EDA → Features → Pipeline → Línea base → Candidatos → Tuning → Selección → Model Registry
```

> **Datos:** anonimizados y sintéticos. No contienen información real de clientes.

## Modelo final

| | |
|---|---|
| **Modelo** | Hist gradient boosting ajustado (`HistGradientBoostingClassifier`, `max_depth=5`, `min_samples_leaf=235`, `learning_rate≈0.035`, `class_weight="balanced"`) |
| **Registro** | `models:/priorizacion-cartera-modelo@champion` (versión 1) |
| **`run_id` de la corrida final** | `aa2fdd0af2bc4b778a923d4676068f09` |
| **PR-AUC en prueba** (21 al 24 de octubre) | **0.520** (regla de negocio actual: 0.239) |
| **PR-AUC en CV por crédito** | 0.459 ± 0.022 |
| **Métrica de negocio** (600 llamadas diarias, 4 días de prueba) | 1,008 pagos en la lista, frente a 603 de la regla actual (+67%) |
| **Challenger** | Random forest ajustado (versión 2, alias `challenger`) |

## Requisitos

- Docker y Docker Compose
- Python **3.12** con [uv](https://docs.astral.sh/uv/). La versión está fijada en `.python-version` (uv la usa automáticamente).
- Una cuenta de Snowflake con la OBT del PSet 2

## 1. Instalar el entorno

```bash
pip install uv
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## 2. Levantar MLflow

```bash
docker compose up -d
```

La interfaz queda en **http://localhost:5000** (en Codespaces: pestaña **Puertos**, puerto 5000).

- **Backend store:** SQLite (`mlflow/mlflow.db`), con parámetros, métricas y tags.
- **Artifact store:** carpeta `mlflow/artifacts`, con los modelos entrenados.
- `restart: unless-stopped`: MLflow se vuelve a encender solo si el equipo o el Codespace se reinicia.

Para apagarlo: `docker compose down`. Las corridas no se pierden, porque quedan en la carpeta `mlflow/`.

## 3. Configurar la conexión a Snowflake

```bash
cp .env.example .env
```

| Variable | Qué es |
|---|---|
| `SNOWFLAKE_ACCOUNT` | Identificador de la cuenta (`ORGANIZACION-CUENTA`) |
| `SNOWFLAKE_USER` | Usuario de servicio (`PSET2_SERVICE`) |
| `SNOWFLAKE_PAT` | Token de acceso programático (PAT) |
| `SNOWFLAKE_ROLE` | Rol con lectura sobre `CARTERA.GOLD` (`PSET2_ROLE`) |
| `SNOWFLAKE_WAREHOUSE` | Warehouse de cómputo (`PSET2_WH`) |
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` |

Para crear un token nuevo, en un SQL File de Snowflake con el rol `ACCOUNTADMIN`:

```sql
ALTER USER PSET2_SERVICE ADD PROGRAMMATIC ACCESS TOKEN PSET3_TOKEN
  ROLE_RESTRICTION = 'PSET2_ROLE'
  DAYS_TO_EXPIRY = 30;
```

El archivo `.env` está en el `.gitignore`: **las credenciales nunca se suben al repositorio.**

## 4. Cómo entrenar y evaluar el modelo

**Datos y preparación** (`src/`):
- `src/datos.py` lee la OBT desde Snowflake (`cargar_obt`), construye el target `paga_7d` (`agregar_target`) y calcula la huella de los datos (`huella`).
- `src/pipeline.py` construye las features, filtra la población (créditos en mora), divide los datos y arma la preparación (`ColumnTransformer`: imputación, `log(1 + x)`, escalado y one-hot), que se ajusta solo con entrenamiento. Se comprueba de punta a punta con:

```bash
python src/pipeline.py
```

**División:** temporal. Entrenamiento con los cortes del 1 al 13 de octubre; prueba con los cortes del 21 al 24. Los 7 días del medio se dejan fuera para que el target del entrenamiento (que mira 7 días hacia adelante) nunca use días del periodo de prueba. La validación cruzada es **por crédito** (`GroupKFold`, 5 pliegues), para que un mismo crédito nunca esté a la vez en entrenamiento y validación.

**Notebooks, en orden** (en VS Code, kernel `.venv`; cada uno con **Restart** y **Run All**):

| # | Notebook | Qué hace | Corridas en MLflow |
|---|---|---|---|
| 1 | `notebooks/01_eda.ipynb` | Exploración de la OBT, target, población y hallazgos | — |
| 2 | `notebooks/02_features.ipynb` | Features derivadas, con evidencia y sin fuga | — |
| 3 | `notebooks/03_baseline_candidatos.ipynb` | 2 líneas base, 4 familias de modelos y la prueba del LLM | 7 |
| 4 | `notebooks/04_tuning.ipynb` | Tuning de los 3 mejores (30 configuraciones cada uno) | 93 |
| 5 | `notebooks/05_seleccion.ipynb` | Ganador, prueba, métrica de negocio, uplift, registro y `docs/corridas.csv` | 1 |

Notas:
- La prueba del LLM (notebook 03) usa la respuesta de Claude pegada en la celda. El prompt queda en `docs/prompt_llm.txt`, y el prompt y la respuesta se guardan como artefactos en MLflow.
- Para repetir el experimento desde cero, borra antes las corridas del experimento `priorizacion-cartera` y el modelo registrado `priorizacion-cartera-modelo`, para que `docs/corridas.csv` no tenga corridas repetidas.

## Estructura

```
pset3-modelo-cartera/
├── docker-compose.yml     # Servidor de MLflow
├── .env.example           # Plantilla de credenciales
├── .python-version        # Python 3.12
├── requirements.txt
├── notebooks/             # 01 a 05, en orden
├── src/
│   ├── datos.py           # Lectura de la OBT, target y huella de los datos
│   └── pipeline.py        # Features, división y preparación (Pipeline + ColumnTransformer)
└── docs/
    ├── corridas.csv       # Exportación de todas las corridas de MLflow (101)
    ├── prompt_llm.txt     # Prompt de la prueba del LLM
    ├── figuras/           # Gráficos de los notebooks, usados en el memo
    └── PSet3_memo_Cardenas_Jacome.pdf
```

# Experimentación y selección del modelo: priorización de cartera

PSet 3 · Ciencia de Datos · Maestría en Ciencia de Datos (USFQ)

Modelo que predice qué créditos **pagan en los próximos 7 días**, para priorizar la gestión de cobranzas. Parte de la OBT construida en el PSet 2 (`CARTERA.GOLD.OBT_CARTERA_DIARIA` en Snowflake), y todas las corridas se registran en MLflow.

```
OBT → EDA → Features → Pipeline → Línea base → Candidatos → Tuning → Selección → Model Registry
```

> **Datos:** anonimizados y sintéticos. No contienen información real de clientes.

## Requisitos

- Docker y Docker Compose
- Python **3.12** (con [uv](https://docs.astral.sh/uv/))
- Una cuenta de Snowflake con la OBT del PSet 2

## 1. Instalar el entorno

```bash
pip install uv
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

## 2. Levantar MLflow

```bash
docker compose up -d
```

La interfaz queda en **http://localhost:5000**.

- **Backend store:** SQLite (`mlflow/mlflow.db`), con parámetros, métricas y tags.
- **Artifact store:** carpeta `mlflow/artifacts`, con los modelos entrenados.

Para apagarlo: `docker compose down`. Las corridas no se pierden, porque quedan en la carpeta `mlflow/`.

## 3. Configurar la conexión a Snowflake

Copia la plantilla y completa tus datos:

```bash
cp .env.example .env
```

| Variable | Qué es |
|---|---|
| `SNOWFLAKE_ACCOUNT` | Identificador de la cuenta (ej. `ABCDEFG-XY12345`) |
| `SNOWFLAKE_USER` | Usuario de Snowflake |
| `SNOWFLAKE_PAT` | Token de acceso personal |
| `SNOWFLAKE_ROLE` | Rol con permiso de lectura sobre `CARTERA.GOLD` |
| `SNOWFLAKE_WAREHOUSE` | Warehouse de cómputo |
| `MLFLOW_TRACKING_URI` |
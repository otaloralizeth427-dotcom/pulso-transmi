# Pulso TransMi — pipeline del equipo

Pronóstico de demanda de pasajeros en 12 estaciones de TransMilenio, a 4
horizontes (+15/+30/+45/+60 min), operado como un ciclo completo de MLOps
para el reto de la Universidad Externado. Ver
[`docs/`](https://github.com/uexternadojz/pulso-transmi-sdk) del SDK oficial
para el contrato de datos, y `guia-metodologica-v1.0.pdf` para el marco del
proyecto.

## Arquitectura

```
API Pulso TransMi  --(collector)-->  Supabase (Postgres)  --(predict)-->  submission
       ^                                     |
       |                                     v
       +-------------- monitor (evalúa, detecta drift) -----------------+
```

- **API Pulso TransMi**: fuente de verdad. `/v1/observations` sólo sirve el
  histórico estático inicial (hasta 2026-09-08); los datos nuevos llegan
  por `/v1/stream/observations`, con cursor. **Esta distinción causó un bug
  real la primera vez que se corrió el pipeline** (ver `src/ingest.py`): usar
  el endpoint equivocado produjo un primer envío con predicciones en cero,
  detectado y corregido antes del cierre del ciclo.
- **Supabase**: memoria operacional. Tablas `stations`, `observations`,
  `context`, `ingestion_state` (cursor del collector), `pipeline_runs`,
  `predictions`, `model_state` (versión activa/champion), `validation_metrics`.
- **GitHub Actions**: `pipeline.yml` corre cada 30 min (collector → predict
  → monitor); `train.yml` es manual/semanal, nunca automático cada ciclo.
- Cada script (`src/ingest.py`, `src/predict.py`, `src/monitor.py`,
  `src/train.py`) puede correr solo — no comparten estado en memoria, todo
  pasa por Postgres o por la API.

## Decisiones de diseño

- **Validación siempre temporal**: `train.py` separa los últimos 7 días como
  validación, nunca una partición aleatoria (mezclar futuro y pasado da
  métricas optimistas que no representan la competencia).
- **Métrica oficial reproducida localmente**: `official_accuracy()` en
  `train.py` es exactamente `100 × max(0, 1 − WAPE)` promediado sin ponderar
  entre las 12 estaciones, para poder comparar candidato vs. baseline con la
  misma vara que usa el leaderboard.
- **Promoción estricta**: un candidato reemplaza al champion sólo si (a)
  supera la accuracy de validación del baseline shift-24h **y** (b) completa
  una inferencia de prueba con valores finitos y no negativos. La novedad
  sola nunca promueve (`train.py::run`).
- **El baseline es un modelo real, no un placeholder**: `demand(t) =
  demand(t-24h)` vive en `model_state` como cualquier otra versión, con su
  propio `model_version` (`baseline-shift24h-v2`). `predict.py` lo usa
  automáticamente mientras sea la versión activa.
- **Reintentos idempotentes**: `predict.py` usa una `Idempotency-Key` estable
  (`auto-{model_version}-{cycle_id}`) — reintentar el mismo ciclo con el
  mismo modelo nunca duplica una entrega.
- **Falla ruidosa, no silenciosa**: si `predict.py` no logra construir las
  48 predicciones exactas que pide el ciclo, no envía nada parcial — corta
  con error explícito.

## Correr localmente

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # completa PTM_API_KEY y SUPABASE_DB_URL
export $(cat .env | xargs)

python src/ingest.py    # trae observaciones nuevas
python src/predict.py   # infiere + envía el ciclo vigente (si hay uno abierto)
python src/monitor.py   # evalúa lo ya resuelto y revisa drift
python src/train.py     # entrena un candidato CatBoost; promueve solo si gana
```

## GitHub Actions — secrets necesarios

En Settings → Secrets and variables → Actions del repo:

- `PTM_API_KEY`: la API key personal del portal (nunca en código ni en el
  repo).
- `SUPABASE_DB_URL`: connection string de Postgres del proyecto Supabase del
  equipo (Project Settings → Database). Opcional para `predict.py`/`ingest.py`
  en el sentido de que el pipeline no truena sin ella, pero sin esto no hay
  trazabilidad ni datos para entrenar.

## Estado actual

- Primer envío oficial aceptado manualmente (evidencia en `pipeline_runs`)
  mientras se activaba esta automatización.
- Champion inicial: `baseline-shift24h-v2`. `train.py` está listo para
  producir y evaluar un candidato CatBoost; se promueve automáticamente la
  primera vez que gane la validación temporal.

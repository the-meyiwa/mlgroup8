# Concrete Strength Predictor

Lives in the https://github.com/the-meyiwa/mlgroup8 repository under `concrete-strength/`.
Every command below is relative to that directory, so `cd concrete-strength` first.

Estimates the compressive strength of a concrete mix (MPa) from eight mixture
proportions, using a scikit-learn regression pipeline trained on the UCI
Concrete Compressive Strength dataset (1,030 laboratory samples).

Built to the three design documents in `docs/`: Design Decisions (DD-01 to
DD-13), System Design, and Data Flow.

## Layout

```
features.json          single canonical feature order, bounds and units (DD-06)
api/                   FastAPI service + model artefact + sidecar metadata
web/                   Next.js single-page form, statically exported
training/              fetch_data.py, train.py, evaluation.md
tests/                 API contract and validation tests
docs/                  design documents and diagrams
```

## Run locally

```bash
make setup      # python deps + npm ci
make data       # download the UCI archive, write data/concrete.csv
make train      # cross-validate candidates, write api/model/model.joblib
make test       # 11 API tests
make api        # uvicorn on :8000
make web        # next dev on :3000
```

Or both containers: `docker compose up --build`.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/api/v1/predict` | Eight numbers in, one strength out |
| GET | `/api/v1/features` | The canonical schema, for the interface |
| GET | `/health` | Liveness and whether the model loaded |
| GET | `/model-info` | Version, algorithm, cross-validated metrics |

```bash
curl -s -X POST http://localhost:8000/api/v1/predict \
  -H 'Content-Type: application/json' \
  -d '{"cement":540,"blast_furnace_slag":0,"fly_ash":0,"water":162,
       "superplasticizer":2.5,"coarse_aggregate":1040,"fine_aggregate":676,"age":28}'
```

```json
{"prediction": 77.3, "unit": "MPa", "model_version": "282638d3c",
 "disclaimer": "This is an estimate ..."}
```

Errors: 422 per-field `errors`, 400 malformed JSON, 429 rate limit with
`Retry-After`, 503 with a correlation id. The model is never called on invalid
input.

## Model

Algorithm is selected by cross-validated RMSE, not by preference (DD-07).
Identical 10-fold splits, seed 42, for every candidate. See
`training/evaluation.md` for the table produced by the current artefact.

The artefact is a single joblib pipeline with preprocessing included. Its
sha256 prefix is the model version returned with every prediction. A JSON
sidecar records metrics, feature order hash, dataset hash, seed and library
versions. If the sidecar disagrees with the schema, the API refuses to start.

## Definition of done for this MVP

- Form, server validation, prediction, disclaimer all working end to end.
- Every rejection path returns the documented status and shape.
- A prediction can be traced to an artefact hash.
- No database, no stored inputs (DD-10).

Out of scope, deliberately: accounts, history, reports, charts, batch
prediction, other concrete types. Each is additive because the endpoint is
stateless and versioned.

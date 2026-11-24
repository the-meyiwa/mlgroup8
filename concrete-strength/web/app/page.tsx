"use client";

import { useMemo, useState } from "react";
import { FEATURES, TARGET, type Feature } from "../lib/features.generated";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

type Values = Record<string, string>;
type FieldErrors = Record<string, string>;

type Prediction = {
  prediction: number;
  unit: string;
  model_version: string;
  disclaimer: string;
};

type ApiError = { field: string; message: string };

const initialValues: Values = Object.fromEntries(FEATURES.map((f) => [f.name, ""]));

function checkClientSide(values: Values): FieldErrors {
  const errors: FieldErrors = {};
  for (const f of FEATURES) {
    const raw = (values[f.name] ?? "").trim();
    if (raw === "") {
      errors[f.name] = `${f.label} is required`;
      continue;
    }
    const n = Number(raw);
    if (!Number.isFinite(n)) {
      errors[f.name] = `${f.label} must be a number`;
      continue;
    }
    if (n < f.min || n > f.max) {
      errors[f.name] = `${f.label} must be between ${f.min} and ${f.max} ${f.unit}`;
    }
  }
  return errors;
}

export default function Page() {
  const [values, setValues] = useState<Values>(initialValues);
  const [errors, setErrors] = useState<FieldErrors>({});
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [busy, setBusy] = useState(false);
  const [banner, setBanner] = useState<string | null>(null);

  const hasErrors = useMemo(() => Object.keys(errors).length > 0, [errors]);

  function update(f: Feature, raw: string) {
    setValues((prev) => ({ ...prev, [f.name]: raw }));
    if (errors[f.name]) {
      setErrors((prev) => {
        const next = { ...prev };
        delete next[f.name];
        return next;
      });
    }
  }

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    setBanner(null);

    const clientErrors = checkClientSide(values);
    if (Object.keys(clientErrors).length > 0) {
      setErrors(clientErrors);
      setPrediction(null);
      return;
    }

    const payload = Object.fromEntries(
      FEATURES.map((f) => [f.name, Number(values[f.name])])
    );

    setBusy(true);
    try {
      const res = await fetch(`${API_BASE}/api/v1/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (res.ok) {
        setPrediction((await res.json()) as Prediction);
        setErrors({});
        return;
      }

      let body: { errors?: ApiError[]; correlation_id?: string } = {};
      try {
        body = await res.json();
      } catch {
        body = {};
      }

      const nextErrors: FieldErrors = {};
      for (const e of body.errors ?? []) {
        if (e.field && FEATURES.some((f) => f.name === e.field)) nextErrors[e.field] = e.message;
      }
      setErrors(nextErrors);
      setPrediction(null);

      if (res.status === 429) {
        setBanner("Too many predictions in a short time. Wait a minute and try again.");
      } else if (res.status === 503) {
        setBanner(
          `The prediction service is unavailable right now. Reference: ${body.correlation_id ?? "unknown"}`
        );
      } else if (res.status === 400) {
        setBanner("That request could not be read. Reload the page and try again.");
      } else if (Object.keys(nextErrors).length === 0) {
        setBanner("Some values were rejected. Check the highlighted fields.");
      }
    } catch {
      setBanner("Could not reach the prediction service. Check your connection and try again.");
    } finally {
      setBusy(false);
    }
  }

  function reset() {
    setValues(initialValues);
    setErrors({});
    setPrediction(null);
    setBanner(null);
  }

  return (
    <main className="page">
      <p className="eyebrow">Concrete Strength Predictor</p>
      <h1>Estimate compressive strength from a mix design</h1>
      <p className="lede">
        Enter the eight mixture proportions below. The model returns one estimated strength in
        megapascals, trained on 1,030 laboratory-tested concrete samples.
      </p>

      <div className="layout">
        <form className="card" onSubmit={submit} noValidate aria-describedby="disclaimer">
          <h2>Mixture proportions</h2>

          {banner && (
            <div className="banner" role="alert">
              {banner}
            </div>
          )}

          <div className="grid">
            {FEATURES.map((f) => {
              const invalid = Boolean(errors[f.name]);
              const id = `field-${f.name}`;
              return (
                <div className={`field${invalid ? " invalid" : ""}`} key={f.name}>
                  <label htmlFor={id}>
                    <span>{f.label}</span>
                    <span className="unit">{f.unit}</span>
                  </label>
                  <input
                    id={id}
                    name={f.name}
                    type="number"
                    inputMode="decimal"
                    step={f.step}
                    min={f.min}
                    max={f.max}
                    value={values[f.name]}
                    onChange={(e) => update(f, e.target.value)}
                    aria-invalid={invalid}
                    aria-describedby={invalid ? `${id}-err` : undefined}
                    placeholder={`${f.min} – ${f.max}`}
                  />
                  {invalid && (
                    <div className="err" id={`${id}-err`}>
                      {errors[f.name]}
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          <div className="actions">
            <button className="primary" type="submit" disabled={busy}>
              {busy ? "Predicting…" : "Predict strength"}
            </button>
            <button className="ghost" type="button" onClick={reset}>
              Clear
            </button>
          </div>
        </form>

        <aside className="result" aria-live="polite">
          <h2>Prediction</h2>
          {prediction ? (
            <>
              <div className="value">
                {prediction.prediction.toFixed(1)}
                <span className="unit-sm">{prediction.unit}</span>
              </div>
              <p className="meta">
                Estimated {TARGET.label.toLowerCase()}
                <br />
                Model version <code>{prediction.model_version}</code>
              </p>
              <p className="disclaimer">{prediction.disclaimer}</p>
            </>
          ) : (
            <p className="placeholder">
              {hasErrors
                ? "Fix the highlighted fields, then predict again."
                : "Fill in all eight values and press Predict. The result appears here with the model version that produced it."}
            </p>
          )}
        </aside>
      </div>

      <footer id="disclaimer">
        The predictor is a regression model (gradient boosting) trained offline on the UCI Concrete
        Compressive Strength dataset. It estimates; it does not test. Results should be confirmed by
        laboratory compression testing before any structural or compliance decision.
      </footer>
    </main>
  );
}

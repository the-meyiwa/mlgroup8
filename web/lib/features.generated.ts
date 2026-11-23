// AUTO-GENERATED from features.json by scripts/gen-features.mjs. Do not edit.

export type Feature = {
  name: string;
  label: string;
  unit: string;
  min: number;
  max: number;
  step: number;
};

export const TARGET = {"name":"strength","unit":"MPa","label":"Compressive strength"} as const;

export const FEATURES: Feature[] = [
  { name: "cement", label: "Cement", unit: "kg/m3", min: 50, max: 800, step: 1 },
  { name: "blast_furnace_slag", label: "Blast furnace slag", unit: "kg/m3", min: 0, max: 600, step: 1 },
  { name: "fly_ash", label: "Fly ash", unit: "kg/m3", min: 0, max: 450, step: 1 },
  { name: "water", label: "Water", unit: "kg/m3", min: 50, max: 350, step: 1 },
  { name: "superplasticizer", label: "Superplasticizer", unit: "kg/m3", min: 0, max: 60, step: 0.1 },
  { name: "coarse_aggregate", label: "Coarse aggregate", unit: "kg/m3", min: 400, max: 1500, step: 1 },
  { name: "fine_aggregate", label: "Fine aggregate", unit: "kg/m3", min: 300, max: 1400, step: 1 },
  { name: "age", label: "Age", unit: "days", min: 1, max: 365, step: 1 },
];

export const FEATURE_NAMES = FEATURES.map((f) => f.name);

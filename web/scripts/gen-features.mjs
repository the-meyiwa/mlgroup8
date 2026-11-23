// Generates web/lib/features.generated.ts from the canonical features.json (DD-06).
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const src = resolve(here, "../../features.json");
const out = resolve(here, "../lib/features.generated.ts");

const schema = JSON.parse(readFileSync(src, "utf8"));
const lines = [
  "// AUTO-GENERATED from features.json by scripts/gen-features.mjs. Do not edit.",
  "",
  "export type Feature = {",
  "  name: string;",
  "  label: string;",
  "  unit: string;",
  "  min: number;",
  "  max: number;",
  "  step: number;",
  "};",
  "",
  "export const TARGET = " + JSON.stringify(schema.target) + " as const;",
  "",
  "export const FEATURES: Feature[] = [",
  ...schema.features.map(
    (f) =>
      "  { name: " + JSON.stringify(f.name) + ", label: " + JSON.stringify(f.label) +
      ", unit: " + JSON.stringify(f.unit) + ", min: " + f.min + ", max: " + f.max +
      ", step: " + f.step + " },"
  ),
  "];",
  "",
  "export const FEATURE_NAMES = FEATURES.map((f) => f.name);",
  "",
];
mkdirSync(dirname(out), { recursive: true });
writeFileSync(out, lines.join("\n"));
console.log("wrote", out, "with", schema.features.length, "features");

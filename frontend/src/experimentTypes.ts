import type { Domain, Job, Spec } from "./types";

export type Prediction = "lower" | "higher" | "unsure";
export type ExperimentVariable =
  | "basis"
  | "grid_level"
  | "functional"
  | "bond_distance"
  | "kmesh"
  | "mesh"
  | "method";
export type ExperimentValue = string | number | number[];
export interface ExperimentTemplate {
  id: string;
  template_version?: string;
  title: string;
  question: string;
  description: string;
  domain?: Domain;
  bases: string[];
  fixed_spec: Spec;
  predictions: { id: Prediction; label: string }[];
  lessons: { title: string; body: string }[];
  variable: ExperimentVariable;
  variable_label?: string;
  values?: ExperimentValue[];
  labels?: string[];
  references?: string[];
  units: Record<string, string>;
}
export interface ExperimentDefinition extends Partial<ExperimentTemplate> {
  expected_specs?: Spec[];
}
export interface ExperimentIntent {
  template_id: string;
  name: string;
  prediction: Prediction;
  request_id: string;
}
export interface ExperimentRow {
  ordinal: number;
  job_id: string;
  basis: string;
  label?: string;
  value?: ExperimentValue;
  status: string;
  comparable: boolean;
  comparison_note?: string | null;
  energy_hartree: number | null;
  delta_previous_hartree: number | null;
  energy_per_atom_hartree?: number | null;
  delta_previous_per_atom_hartree?: number | null;
  basis_functions: number | null;
  iteration_count: number | null;
  elapsed_seconds: number | null;
  peak_memory_mb: number | null;
  error: string | null;
}
export interface Experiment {
  id: string;
  name: string;
  template_id: string;
  prediction: Prediction;
  request_id: string;
  created_at: string;
  cancel_requested: boolean;
  base_spec: Spec;
  bases: string[];
  definition?: ExperimentDefinition;
  domain?: Domain;
  variable?: ExperimentVariable;
  variable_label?: string;
  labels?: string[];
  values?: ExperimentValue[];
  status:
    "queued" | "running" | "completed" | "completed_with_issues" | "cancelled";
  completed_jobs: number;
  successful_jobs: number;
  total_jobs: number;
  active_job_id: string | null;
  rows: ExperimentRow[];
  analysis: {
    valid_count: number;
    total_count: number;
    complete: boolean;
    comparable: boolean;
    energy_trend: string | null;
    last_step_delta_hartree: number | null;
    total_change_hartree: number | null;
    messages: { kind: "observation" | "limit" | "next"; text: string }[];
  };
  jobs: Job[];
}
export const EXPERIMENT_TERMINAL = new Set([
  "completed",
  "completed_with_issues",
  "cancelled",
]);

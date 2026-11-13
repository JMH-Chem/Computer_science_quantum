export interface Atom {
  symbol: string;
  position: number[];
}
export type Domain = "molecule" | "crystal";
export type Task =
  | "single_point"
  | "optimize"
  | "tda"
  | "tddft"
  | "bands"
  | "educational_rhf"
  | "volume";
export interface VolumeRequest {
  kind: "density" | "orbital" | "nto_hole" | "nto_particle";
  index: number;
  spin: number;
  grid: number;
}
export interface Spec {
  name: string;
  domain: Domain;
  task: Task;
  atoms: Atom[];
  lattice: number[][] | null;
  charge: number;
  spin: number;
  method: "HF" | "DFT";
  basis: string | Record<string, string>;
  ecp: string | Record<string, string> | null;
  functional: string;
  restricted: boolean;
  kmesh: number[];
  pseudo: string;
  grid_level: number;
  mesh: number[];
  max_cycle: number;
  conv_tol: number;
  nstates: number;
  max_steps: number;
  threads: number;
  memory_mb: number;
  band_points: number;
  parent_job_id?: string | null;
  volume?: VolumeRequest | null;
}
export interface Example {
  id: string;
  title: string;
  description: string;
  spec: Spec;
}
export interface Iteration {
  iteration: number;
  energy_hartree: number;
  delta_energy?: number;
  gradient_norm?: number;
  density_change?: number;
  optimization_step?: number;
}
export interface OptimizationStep {
  step: number;
  energy_hartree: number;
  atoms: Atom[];
}
export interface Excitation {
  converged?: boolean;
  state: number;
  energy_ev: number;
  oscillator_strength: number;
  nto_weights?: number[];
}
export interface Result {
  converged: boolean;
  energy_hartree?: number;
  atoms?: Atom[];
  lattice?: number[][] | null;
  engine?: { name: string; version: string };
  orbital_energies_hartree?: number[] | number[][];
  occupations?: number[] | number[][];
  dipole_debye?: number[];
  excitations?: Excitation[];
  optimization?: OptimizationStep[];
  artifact?: string;
  warnings?: string[];
  bands?: {
    distance: number[];
    break_indices?: number[];
    n_occupied?: number;
    energies_ev: number[][];
    labels: { distance: number; label: string }[];
    reference: string;
    gap_ev?: number;
  };
  education?: Record<string, unknown>;
  metadata?: Record<string, unknown>;
  [key: string]: unknown;
}
export interface Job {
  id: string;
  name?: string;
  status: string;
  spec: Spec;
  result?: Result | null;
  error?: string | null;
  created_at?: string;
  updated_at?: string;
}
export interface JobEvent {
  seq: number;
  type: string;
  data: Record<string, unknown>;
  timestamp?: string;
}
export interface CatalogItem {
  name?: string;
  id?: string;
  label?: string;
  description?: string;
  status?: string;
  family?: string;
  [key: string]: unknown;
}
export interface Lesson {
  id?: string;
  title: string;
  body?: string;
  content?: string;
  source?: string;
  url?: string;
  [key: string]: unknown;
}
export interface Catalog {
  functionals: (string | CatalogItem)[];
  bases: (string | CatalogItem)[];
  methods: (string | CatalogItem)[];
  lessons: Lesson[];
}
export const TERMINAL = new Set([
  "succeeded",
  "unconverged",
  "failed",
  "cancelled",
  "interrupted",
]);
export const defaultSpec: Spec = {
  name: "물 분자 · HF",
  domain: "molecule",
  task: "single_point",
  atoms: [
    { symbol: "O", position: [0, 0, 0] },
    { symbol: "H", position: [0, -0.757, 0.587] },
    { symbol: "H", position: [0, 0.757, 0.587] },
  ],
  lattice: null,
  charge: 0,
  spin: 0,
  method: "HF",
  basis: "sto-3g",
  ecp: null,
  functional: "PBE",
  restricted: true,
  kmesh: [1, 1, 1],
  pseudo: "gth-pbe",
  grid_level: 2,
  mesh: [15, 15, 15],
  max_cycle: 80,
  conv_tol: 1e-9,
  nstates: 5,
  max_steps: 50,
  threads: 2,
  memory_mb: 2000,
  band_points: 40,
};
export const statusLabels: Record<string, string> = {
  queued: "대기 중",
  running: "계산 중",
  succeeded: "수렴 완료",
  unconverged: "미수렴",
  failed: "실패",
  cancelled: "취소됨",
  interrupted: "중단됨",
};
export const taskLabels: Record<string, string> = {
  single_point: "단일점 에너지",
  optimize: "구조 최적화",
  tda: "TDA · 여기 상태",
  tddft: "TDDFT · 여기 상태",
  bands: "밴드 구조",
  educational_rhf: "교육용 RHF",
  volume: "3D 격자 생성",
};
export const atomText = (atoms: Atom[]) =>
  atoms
    .map(
      (a) => `${a.symbol}  ${a.position.map((n) => n.toFixed(6)).join("  ")}`,
    )
    .join("\n");
export const xyzText = (atoms: Atom[]) =>
  `${atoms.length}\nQuantum Lab · Angstrom\n${atomText(atoms)}\n`;
export const itemName = (item: string | CatalogItem) =>
  typeof item === "string"
    ? item
    : String(item.name ?? item.id ?? item.label ?? "");

export interface Atom {
  symbol: string;
  position: number[];
}
export type Domain = "molecule" | "crystal";
export type Method =
  | "HF"
  | "DFT"
  | "MP2"
  | "CCSD"
  | "CCSD(T)"
  | "CISD"
  | "FCI"
  | "CASCI"
  | "CASSCF"
  | "NEVPT2";
export const CORRELATED_METHODS: Method[] = [
  "MP2",
  "CCSD",
  "CCSD(T)",
  "CISD",
  "FCI",
  "CASCI",
  "CASSCF",
  "NEVPT2",
];
export type Task =
  | "single_point"
  | "optimize"
  | "tda"
  | "tddft"
  | "tdhf"
  | "frequencies"
  | "eomccsd"
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
  method: Method;
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
  frozen_orbitals?: number;
  active_electrons?: number;
  active_orbitals?: number;
  ir_intensities?: boolean;
  temperature_kelvin?: number;
  pressure_pa?: number;
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
  symmetry?: string | null;
  symmetry_group?: string | null;
  residual_norm?: number | null;
}
export interface RootSearch {
  strategy: string;
  group: string | null;
  complete: boolean;
  scope: string;
  blocks: Record<string, unknown>[];
}
export interface Result {
  converged: boolean;
  energy_hartree?: number;
  atoms?: Atom[];
  lattice?: number[][] | null;
  engine?: { name: string; version: string };
  orbital_energies_hartree?: (number | null)[] | (number | null)[][];
  occupations?: number[] | number[][];
  dipole_debye?: number[];
  reference_energy_hartree?: number;
  correlation_energy_hartree?: number;
  density_reference?: string;
  orbital_reference?: string;
  dipole_reference?: string;
  correlation?: CorrelationResult;
  eom?: EOMResult;
  properties?: MolecularProperties;
  vibrations?: VibrationalResult;
  sampled_dos?: SampledDOS;
  sampled_dos_unavailable_reason?: string;
  crystal_properties?: CrystalProperties;
  sampled_band_edges?: SampledBandEdges;
  excitations?: Excitation[];
  optimization?: OptimizationStep[];
  artifact?: string;
  warnings?: string[];
  bands?: {
    distance: number[];
    break_indices?: number[];
    n_occupied?: number;
    energies_ev: (number | null)[][];
    retained_bands_per_kpoint?: number[];
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
export interface CorrelationIteration {
  iteration: number;
  energy_hartree: number;
  delta_energy?: number;
  residual_norm?: number;
  stage?: string;
}
export interface EOMResult {
  reference: string;
  spin: string;
  converged: boolean;
  states: {
    state: number;
    energy_ev: number;
    converged: boolean;
    residual_norm: number;
    symmetry?: string | null;
    symmetry_group?: string | null;
  }[];
  root_search?: RootSearch;
  requested_states: number;
  residual_tolerance: number;
  wavefunction_artifact: string;
  notes: string[];
}
export interface CorrelationResult {
  method: string;
  reference: string;
  converged: boolean;
  reference_energy_hartree: number;
  total_energy_hartree: number;
  correlation_energy_hartree: number;
  frozen_orbitals: number;
  active_space: {
    electrons: number;
    orbitals: number;
    core_orbitals: number;
    initial_orbital_indices_1based?: number[];
  } | null;
  diagnostics: Record<string, unknown>;
  natural_occupations?: number[];
  natural_occupations_space?: string;
  iterations: CorrelationIteration[];
}
export interface MolecularProperties {
  reference_model: string;
  mulliken_charges: number[];
  lowdin_charges: number[];
  dipole_debye: number[];
  dipole_magnitude_debye: number;
  spin_square?: number;
  spin_multiplicity?: number;
  units: { charges: string; dipole: string };
  warnings: string[];
}
export interface VibrationalMode {
  index: number;
  frequency_cm1: number;
  imaginary: boolean;
  reduced_mass_amu: number;
  displacements: number[][];
  ir_intensity_km_mol: number | null;
}
export interface VibrationalResult {
  converged: boolean;
  frequencies_cm1: number[];
  modes: VibrationalMode[];
  masses_amu: number[];
  gradient_max_hartree_per_bohr: number;
  stationary: boolean;
  stationarity_threshold_hartree_per_bohr: number;
  imaginary_modes: number;
  zero_point_energy_hartree: number | null;
  thermochemistry: null | {
    model: string;
    temperature_kelvin: number;
    pressure_pa: number;
    internal_energy_hartree: number;
    enthalpy_hartree: number;
    gibbs_energy_hartree: number;
    entropy_hartree_per_kelvin: number;
    heat_capacity_cv_hartree_per_kelvin: number;
    heat_capacity_cp_hartree_per_kelvin: number;
    rotational_symmetry_number: number;
    frequency_scaling_factor: number;
  };
  ir: {
    requested: boolean;
    computed: boolean;
    finite_difference_step_bohr: number;
    displaced_calculations: number;
  };
  warnings: string[];
  artifacts: string[];
}
export interface SampledDOS {
  energy_ev: number[];
  density_states_per_ev: number[];
  occupied_density_states_per_ev: number[];
  width_ev: number;
  kpoint_count: number;
  spin_degeneracy: number;
  kpoint_weights: number[];
  reference: string;
  reference_energy_hartree: number;
  state_integral: number;
  expected_states: number;
  electron_integral: number;
  expected_electrons: number;
  units: string;
  broadening: string;
  sample: string;
  limitations: string[];
}
export interface CrystalProperties {
  energy_per_atom_hartree: number;
  volume_angstrom3: number;
  density_g_cm3: number;
  atom_count: number;
  mass_convention: string;
}
export interface SampledBandEdges {
  sample: string;
  indirect_gap_ev: number;
  direct_gap_ev: number;
  vbm: {
    energy_hartree: number;
    energy_ev: number;
    kpoint_scaled: number[];
    source: string;
    index: number;
    band_index: number;
  };
  cbm: {
    energy_hartree: number;
    energy_ev: number;
    kpoint_scaled: number[];
    source: string;
    index: number;
    band_index: number;
  };
  direct_gap_location: {
    kpoint_scaled: number[];
    source: string;
    index: number;
  };
  classification: string;
  reference: string;
  gap_kind: string;
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
  frozen_orbitals: 0,
  active_electrons: 2,
  active_orbitals: 2,
  ir_intensities: false,
  temperature_kelvin: 298.15,
  pressure_pa: 101325,
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
  tdhf: "TDHF · HF 여기 상태",
  frequencies: "진동수 · 조화 진동 분석",
  eomccsd: "EOM-CCSD · singlet 여기 상태",
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

import type { Job } from "./types";

export function finalEnergy(job: Job): number | null {
  const energy = job.result?.energy_hartree;
  return job.status === "succeeded" &&
    job.result?.converged === true &&
    typeof energy === "number" &&
    Number.isFinite(energy)
    ? energy
    : null;
}

// Conservative comparison of identical systems and electronic models. Numerical
// choices (basis, grid, k mesh, convergence thresholds) remain visible in the table.
export function comparisonProblem(
  reference: Job,
  candidate: Job,
): string | null {
  if (finalEnergy(reference) === null || finalEnergy(candidate) === null)
    return "최종 수렴 에너지 없음";
  if (reference.spec.domain !== candidate.spec.domain)
    return "계산 대상 종류가 다름";
  if (
    reference.spec.charge !== candidate.spec.charge ||
    reference.spec.spin !== candidate.spec.spin ||
    reference.spec.restricted !== candidate.spec.restricted
  )
    return "전하·스핀·제한 조건이 다름";
  if (
    reference.spec.method !== candidate.spec.method ||
    (reference.spec.method === "DFT" &&
      reference.spec.functional.toUpperCase() !==
        candidate.spec.functional.toUpperCase())
  )
    return "방법·함수형이 다름: 같은 모델의 수렴 비교에서 제외";
  if (
    JSON.stringify(reference.spec.ecp ?? null) !==
      JSON.stringify(candidate.spec.ecp ?? null) ||
    (reference.spec.domain === "crystal" &&
      reference.spec.pseudo.toLowerCase() !==
        candidate.spec.pseudo.toLowerCase())
  )
    return "핵심 전자 처리(ECP·가상퍼텐셜)가 다름";
  if (
    (reference.spec.frozen_orbitals ?? 0) !==
    (candidate.spec.frozen_orbitals ?? 0)
  )
    return "동결 오비탈 조건이 다름";
  if (
    ["CASCI", "CASSCF", "NEVPT2"].includes(reference.spec.method) &&
    ((reference.spec.active_electrons ?? 2) !==
      (candidate.spec.active_electrons ?? 2) ||
      (reference.spec.active_orbitals ?? 2) !==
        (candidate.spec.active_orbitals ?? 2))
  )
    return "활성 공간이 다름";
  const validVector = (value: unknown): value is number[] =>
    Array.isArray(value) &&
    value.length === 3 &&
    value.every((entry) => typeof entry === "number" && Number.isFinite(entry));
  const validAtoms = (value: unknown) =>
    Array.isArray(value) &&
    value.length > 0 &&
    value.every(
      (atom) =>
        atom && typeof atom.symbol === "string" && validVector(atom.position),
    );
  const left = reference.result?.atoms;
  const right = candidate.result?.atoms;
  if (!left || !right || !validAtoms(left) || !validAtoms(right))
    return "유효한 최종 구조 기록 없음";
  if (
    left.length !== right.length ||
    left.some(
      (atom, i) =>
        atom.symbol !== right[i]?.symbol ||
        atom.position.some(
          (value, axis) => Math.abs(value - right[i].position[axis]) > 1e-6,
        ),
    )
  )
    return "최종 구조·원자 순서가 다름 (좌표 허용오차 10⁻⁶ Å)";
  const a = reference.result?.lattice ?? reference.spec.lattice;
  const b = candidate.result?.lattice ?? candidate.spec.lattice;
  const validLattice = (value: unknown) =>
    Array.isArray(value) && value.length === 3 && value.every(validVector);
  if (
    reference.spec.domain === "crystal" &&
    (!validLattice(a) || !validLattice(b))
  )
    return "유효한 격자 기록 없음";
  if (
    !!a !== !!b ||
    a?.some((vector, i) =>
      vector.some(
        (value, axis) => Math.abs(value - (b?.[i]?.[axis] ?? NaN)) > 1e-6,
      ),
    )
  )
    return "격자가 다름";
  if (
    !reference.result?.engine?.version ||
    !candidate.result?.engine?.version ||
    reference.result.engine.name !== candidate.result.engine.name ||
    reference.result.engine.version !== candidate.result.engine.version
  )
    return "엔진·버전이 다르거나 기록 없음";
  const provenance = (job: Job) =>
    job.result?.provenance as Record<string, unknown> | undefined;
  const keys =
    reference.spec.task === "educational_rhf"
      ? ["pyscf", "numpy", "integral_provider"]
      : [
          "basis_provider",
          "basis_provider_version",
          "numpy_version",
          ...(reference.spec.method === "DFT" ? ["libxc_version"] : []),
        ];
  for (const key of keys) {
    const a = provenance(reference)?.[key],
      b = provenance(candidate)?.[key];
    if (typeof a !== "string" || !a || typeof b !== "string" || !b)
      return "필요한 라이브러리 버전 기록 없음";
    if (a !== b) return "계산 라이브러리 버전이 다름";
  }
  return null;
}

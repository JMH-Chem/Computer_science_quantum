import { useRef, useState } from "react";
import {
  ChevronDown,
  FileUp,
  FlaskConical,
  Layers3,
  Play,
  Search,
  Settings2,
} from "lucide-react";
import { api } from "./api";
import type { Atom, Catalog, CatalogItem, Example, Spec, Task } from "./types";
import { atomText, itemName, taskLabels } from "./types";

export default function Settings({
  spec,
  setSpec,
  examples,
  catalog,
  expert,
  onRun,
  busy,
  onError,
}: {
  spec: Spec;
  setSpec: (s: Spec) => void;
  examples: Example[];
  catalog: Catalog;
  expert: boolean;
  onRun: (s: Spec) => void;
  busy: boolean;
  onError: (s: string) => void;
}) {
  const [coordinates, setCoordinates] = useState(atomText(spec.atoms));
  const [lattice, setLattice] = useState(JSON.stringify(spec.lattice));
  const [basisSearch, setBasisSearch] = useState(""),
    [bases, setBases] = useState<CatalogItem[]>([]),
    [ecp, setEcp] = useState(spec.ecp ? JSON.stringify(spec.ecp) : ""),
    [elementBasis, setElementBasis] = useState("");
  const [importing, setImporting] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const update = (partial: Partial<Spec>) => setSpec({ ...spec, ...partial });
  const load = (s: Spec) => {
    setSpec(s);
    setCoordinates(atomText(s.atoms));
    setLattice(JSON.stringify(s.lattice));
    setEcp(s.ecp ? JSON.stringify(s.ecp) : "");
    setElementBasis(typeof s.basis === "object" ? JSON.stringify(s.basis) : "");
  };
  const [loadedSpec, setLoadedSpec] = useState(spec);
  if (spec !== loadedSpec) {
    setLoadedSpec(spec);
    if (JSON.stringify(spec.atoms) !== JSON.stringify(loadedSpec.atoms))
      setCoordinates(atomText(spec.atoms));
    if (JSON.stringify(spec.lattice) !== JSON.stringify(loadedSpec.lattice))
      setLattice(JSON.stringify(spec.lattice));
    if (JSON.stringify(spec.ecp) !== JSON.stringify(loadedSpec.ecp))
      setEcp(
        spec.ecp
          ? typeof spec.ecp === "string"
            ? spec.ecp
            : JSON.stringify(spec.ecp)
          : "",
      );
    if (JSON.stringify(spec.basis) !== JSON.stringify(loadedSpec.basis))
      setElementBasis(
        typeof spec.basis === "object" ? JSON.stringify(spec.basis) : "",
      );
  }
  const domain = (value: "molecule" | "crystal") => {
    const ex = examples.find((e) => e.spec.domain === value);
    if (ex) load(ex.spec);
    else
      update({
        domain: value,
        task: value === "crystal" ? "bands" : "single_point",
        method: value === "crystal" ? "DFT" : "HF",
        basis: value === "crystal" ? "gth-szv" : "sto-3g",
        lattice:
          value === "crystal"
            ? [
                [0, 1.7834, 1.7834],
                [1.7834, 0, 1.7834],
                [1.7834, 1.7834, 0],
              ]
            : null,
      });
  };
  async function parseFile(file: File) {
    setImporting(true);
    try {
      const format = file.name.toLowerCase().endsWith(".cif") ? "cif" : "xyz";
      const parsed = await api<{
        atoms: Atom[];
        lattice: number[][] | null;
        domain: "molecule" | "crystal";
      }>("/structures/parse", {
        method: "POST",
        body: JSON.stringify({ format, content: await file.text() }),
      });
      load({
        ...spec,
        ...parsed,
        name: file.name,
        task: parsed.domain === "crystal" ? "bands" : "single_point",
        method: parsed.domain === "crystal" ? "DFT" : "HF",
        basis: parsed.domain === "crystal" ? "gth-szv" : "sto-3g",
        charge: 0,
        spin: 0,
        restricted: true,
        functional: "PBE",
        ecp: null,
        kmesh: [1, 1, 1],
      });
    } catch (e) {
      onError(String(e));
    } finally {
      setImporting(false);
    }
  }
  function collect() {
    try {
      const atoms = coordinates
        .trim()
        .split("\n")
        .filter((s) => s.trim())
        .map((line) => {
          const parts = line.trim().split(/\s+/);
          if (
            parts.length !== 4 ||
            parts.slice(1).some((n) => !Number.isFinite(Number(n)))
          )
            throw new Error("좌표는 원소 x y z 형식으로 입력하세요.");
          return { symbol: parts[0], position: parts.slice(1).map(Number) };
        });
      const result = {
        ...spec,
        atoms,
        lattice: spec.domain === "crystal" ? JSON.parse(lattice) : null,
        ecp: ecp.trim()
          ? ecp.trim().startsWith("{")
            ? JSON.parse(ecp)
            : ecp.replace(/^"|"$/g, "")
          : null,
        basis: elementBasis.trim() ? JSON.parse(elementBasis) : spec.basis,
      };
      if (!atoms.length) throw new Error("원자를 입력하세요.");
      setSpec(result);
      onRun(result);
    } catch (e) {
      onError(e instanceof Error ? e.message : String(e));
    }
  }
  async function searchBasis() {
    try {
      const answer = await api<CatalogItem[]>(
        `/catalog/bases?q=${encodeURIComponent(basisSearch)}&elements=${encodeURIComponent([...new Set(spec.atoms.map((a) => a.symbol))].join(","))}`,
      );
      setBases(answer.slice(0, 30));
    } catch (e) {
      onError(String(e));
    }
  }
  const molecularTasks: Task[] = [
    "single_point",
    "optimize",
    "tda",
    "tddft",
    "educational_rhf",
  ];
  const selectedFunctional = catalog.functionals.find(
    (item) => itemName(item).toLowerCase() === spec.functional.toLowerCase(),
  );
  const functionalInfo =
    typeof selectedFunctional === "object" ? selectedFunctional : null;
  const selectedBasis = catalog.bases.find(
    (item) =>
      typeof spec.basis === "string" &&
      itemName(item).toLowerCase() === spec.basis.toLowerCase(),
  );
  const basisInfo = typeof selectedBasis === "object" ? selectedBasis : null;
  const supportText = (item: CatalogItem | null) =>
    item?.status === "verified"
      ? "검증 기록 있음 · 해당 조건 확인"
      : item?.status === "unsupported"
        ? "현재 미지원"
        : "실행 가능 · 해당 조합 미검증";
  return (
    <aside className="settings-panel">
      <div className="panel-label">
        <Settings2 size={15} /> 계산 설정 <span>SETUP</span>
      </div>
      <div className="segmented domain-select">
        <button
          className={spec.domain === "molecule" ? "active" : ""}
          onClick={() => domain("molecule")}
        >
          <FlaskConical size={16} />
          분자
        </button>
        <button
          className={spec.domain === "crystal" ? "active" : ""}
          onClick={() => domain("crystal")}
        >
          <Layers3 size={16} />
          재료
        </button>
      </div>
      <label className="field">
        예제로 시작하기
        <select
          aria-label="예제 선택"
          value=""
          onChange={(e) => {
            const ex = examples.find((x) => x.id === e.target.value);
            if (ex) load(ex.spec);
          }}
        >
          <option value="" disabled>
            예제 선택
          </option>
          {examples
            .filter((e) => e.spec.domain === spec.domain)
            .map((e) => (
              <option key={e.id} value={e.id}>
                {e.title}
              </option>
            ))}
        </select>
      </label>
      <label className="field">
        계산 이름
        <input
          value={spec.name}
          onChange={(e) => update({ name: e.target.value })}
        />
      </label>
      <div className="section-line">
        <span>01</span> 구조{" "}
        <button
          className="text-button"
          onClick={() => input.current?.click()}
          disabled={importing}
        >
          <FileUp size={13} />
          {importing ? "읽는 중" : "XYZ / CIF"}
        </button>
        <input
          ref={input}
          type="file"
          accept=".xyz,.cif"
          hidden
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) void parseFile(file);
            e.target.value = "";
          }}
        />
      </div>
      <label className="field">
        원자 좌표 <small>Å · 원소 x y z</small>
        <textarea
          aria-label="원자 좌표"
          className="coordinate-input"
          spellCheck={false}
          rows={5}
          value={coordinates}
          onChange={(e) => setCoordinates(e.target.value)}
        />
      </label>
      {spec.domain === "crystal" && (
        <label className="field">
          격자 벡터 <small>Å · JSON 3 × 3</small>
          <textarea
            aria-label="격자 벡터"
            rows={3}
            value={lattice}
            onChange={(e) => setLattice(e.target.value)}
          />
        </label>
      )}
      {spec.domain === "molecule" && (
        <div className="field-row">
          <label className="field">
            전하
            <input
              type="number"
              aria-label="전하"
              value={spec.charge}
              onChange={(e) => update({ charge: Number(e.target.value) })}
            />
          </label>
          <label className="field">
            스핀 <small>Nα − Nβ</small>
            <input
              type="number"
              min={0}
              aria-label="스핀"
              value={spec.spin}
              onChange={(e) =>
                update({
                  spin: Number(e.target.value),
                  restricted:
                    Number(e.target.value) === 0 ? spec.restricted : false,
                })
              }
            />
          </label>
        </div>
      )}
      <div className="section-line">
        <span>02</span> 계산 방법
      </div>
      <label className="field">
        작업
        <select
          aria-label="계산 작업"
          value={spec.task}
          onChange={(e) => {
            const task = e.target.value as Task;
            update({
              task,
              ...(["tda", "tddft"].includes(task)
                ? { method: "DFT" as const, spin: 0, restricted: true }
                : task === "educational_rhf"
                  ? { method: "HF" as const, spin: 0, restricted: true }
                  : {}),
            });
          }}
        >
          {(spec.domain === "molecule"
            ? molecularTasks
            : (["single_point", "bands"] as Task[])
          ).map((task) => (
            <option key={task} value={task}>
              {taskLabels[task]}
            </option>
          ))}
        </select>
      </label>
      <div className="field-row">
        <label className="field">
          방법론
          <select
            aria-label="방법론"
            value={spec.method}
            disabled={
              spec.domain === "crystal" ||
              ["tda", "tddft", "educational_rhf"].includes(spec.task)
            }
            onChange={(e) => update({ method: e.target.value as "HF" | "DFT" })}
          >
            <option value="HF">Hartree–Fock</option>
            <option value="DFT">DFT</option>
          </select>
        </label>
        <label className="field">
          기저 집합
          <select
            aria-label="기저 집합"
            value={typeof spec.basis === "string" ? spec.basis : ""}
            onChange={(e) => {
              update({ basis: e.target.value });
              setElementBasis("");
            }}
          >
            {(spec.domain === "crystal"
              ? ["gth-szv", "gth-dzvp"]
              : [
                  ...new Set([
                    "sto-3g",
                    "def2-svp",
                    "def2-tzvp",
                    ...(typeof spec.basis === "string" ? [spec.basis] : []),
                    ...catalog.bases
                      .map(itemName)
                      .filter((n) => !n.toLowerCase().startsWith("gth-"))
                      .slice(0, 20),
                  ]),
                ]
            ).map((name) => (
              <option value={name} key={name}>
                {name}
              </option>
            ))}
          </select>
        </label>
      </div>
      <div className="catalog-status">
        기저: {supportText(basisInfo)}
        {basisInfo?.description && <small>{basisInfo.description}</small>}
      </div>
      {spec.method === "DFT" && (
        <label className="field">
          교환·상관 함수형
          <input
            aria-label="함수형"
            list="functional-list"
            value={spec.functional}
            disabled={spec.domain === "crystal"}
            onChange={(e) => update({ functional: e.target.value })}
          />
          <datalist id="functional-list">
            {[
              ...new Set([
                "PBE",
                "PBE0",
                "B3LYP",
                ...catalog.functionals.map(itemName),
              ]),
            ].map((name) => (
              <option key={name} value={name} />
            ))}
          </datalist>
          <small>
            {functionalInfo?.family} · {supportText(functionalInfo)}
          </small>
          {functionalInfo && (
            <details className="catalog-detail">
              <summary>함수형 설명과 출처</summary>
              <p>{functionalInfo.description}</p>
              {Array.isArray(functionalInfo.supported_tasks) && (
                <p>
                  지원 작업:{" "}
                  {functionalInfo.supported_tasks
                    .map((t) => taskLabels[String(t)] ?? String(t))
                    .join(" · ")}
                </p>
              )}
              {Array.isArray(functionalInfo.references) && (
                <ul>
                  {[...new Set(functionalInfo.references.map(String))].map(
                    (reference) => (
                      <li key={reference}>{reference}</li>
                    ),
                  )}
                </ul>
              )}
              {typeof functionalInfo.reference === "string" && (
                <a
                  href={functionalInfo.reference}
                  target="_blank"
                  rel="noreferrer"
                >
                  공식 문서
                </a>
              )}
            </details>
          )}
          <small>카탈로그 등재와 현재 작업의 지원 여부는 다릅니다.</small>
        </label>
      )}
      {spec.domain === "crystal" && (
        <>
          <label className="field">
            가상퍼텐셜
            <input value={spec.pseudo} readOnly />
          </label>
          <label className="field">
            k점 격자
            <select
              aria-label="k점 격자"
              value={spec.kmesh.join(",")}
              onChange={(e) =>
                update({ kmesh: e.target.value.split(",").map(Number) })
              }
            >
              <option value="1,1,1">Γ · 1 × 1 × 1 (빠른 예제)</option>
              <option value="2,2,2">2 × 2 × 2</option>
              <option value="3,3,3">3 × 3 × 3</option>
            </select>
          </label>
        </>
      )}
      {["tda", "tddft"].includes(spec.task) && (
        <label className="field">
          계산할 여기 상태 수
          <input
            type="number"
            min={1}
            max={20}
            value={spec.nstates}
            onChange={(e) => update({ nstates: Number(e.target.value) })}
          />
        </label>
      )}
      {expert && (
        <details className="advanced" open>
          <summary>
            전문 설정 <ChevronDown size={14} />
          </summary>
          {spec.domain === "molecule" && (
            <>
              <label className="check-field">
                <input
                  type="checkbox"
                  checked={spec.restricted}
                  onChange={(e) => update({ restricted: e.target.checked })}
                />
                제한 궤도 (restricted)
              </label>
              <label className="field">
                기저 검색 (Basis Set Exchange)
                <div className="search-input">
                  <input
                    aria-label="기저 검색"
                    value={basisSearch}
                    onChange={(e) => setBasisSearch(e.target.value)}
                    placeholder="def2, aug-cc…"
                  />
                  <button
                    className="icon-button"
                    aria-label="기저 검색 실행"
                    onClick={() => void searchBasis()}
                  >
                    <Search size={15} />
                  </button>
                </div>
              </label>
              {bases.length > 0 && (
                <div className="search-results">
                  {bases.map((b) => (
                    <button
                      key={itemName(b)}
                      onClick={() => {
                        update({ basis: itemName(b) });
                        setBases([]);
                        setElementBasis("");
                      }}
                    >
                      {itemName(b)}
                      <small>{String(b.status ?? "미검증")}</small>
                    </button>
                  ))}
                </div>
              )}
              <label className="field">
                원소별 기저 <small>선택 · JSON</small>
                <textarea
                  value={elementBasis}
                  onChange={(e) => setElementBasis(e.target.value)}
                  placeholder={'{"O":"def2-svp","H":"sto-3g"}'}
                  rows={2}
                />
              </label>
              <label className="field">
                ECP <small>선택 · 이름 또는 JSON</small>
                <input
                  value={ecp}
                  onChange={(e) => setEcp(e.target.value)}
                  placeholder="원소·기저와 호환 여부 확인"
                />
              </label>
            </>
          )}
          <div className="field-row">
            <label className="field">
              SCF 최대 반복
              <input
                aria-label="SCF 최대 반복"
                type="number"
                min={1}
                max={500}
                value={spec.max_cycle}
                onChange={(e) => update({ max_cycle: Number(e.target.value) })}
              />
            </label>
            <label className="field">
              에너지 허용오차
              <input
                type="number"
                step="any"
                value={spec.conv_tol}
                onChange={(e) => update({ conv_tol: Number(e.target.value) })}
              />
            </label>
          </div>
          {spec.task === "optimize" && (
            <label className="field">
              최적화 최대 단계
              <input
                type="number"
                min={1}
                value={spec.max_steps}
                onChange={(e) => update({ max_steps: Number(e.target.value) })}
              />
            </label>
          )}
          {spec.domain === "crystal" ? (
            <>
              <label className="field">
                수치 격자 · 각 방향
                <input
                  aria-label="수치 격자"
                  type="number"
                  min={9}
                  max={99}
                  step={2}
                  value={spec.mesh[0]}
                  onChange={(e) =>
                    update({ mesh: Array(3).fill(Number(e.target.value)) })
                  }
                />
              </label>
              <label className="field">
                밴드 경로 점 수
                <input
                  type="number"
                  min={10}
                  max={200}
                  value={spec.band_points}
                  onChange={(e) =>
                    update({ band_points: Number(e.target.value) })
                  }
                />
              </label>
            </>
          ) : (
            <label className="field">
              DFT 적분 격자 수준
              <input
                type="number"
                min={0}
                max={5}
                value={spec.grid_level}
                onChange={(e) => update({ grid_level: Number(e.target.value) })}
              />
            </label>
          )}
          <div className="field-row">
            <label className="field">
              CPU 스레드
              <input
                type="number"
                min={1}
                max={8}
                value={spec.threads}
                onChange={(e) => update({ threads: Number(e.target.value) })}
              />
            </label>
            <label className="field">
              메모리 예산 · MB
              <input
                type="number"
                min={256}
                value={spec.memory_mb}
                step={256}
                onChange={(e) => update({ memory_mb: Number(e.target.value) })}
              />
            </label>
          </div>
        </details>
      )}
      <div className="setup-note">
        {spec.domain === "crystal"
          ? "중성 3D 비금속 결정 · Gaussian 궤도 기저"
          : "작은 분자부터 조건을 비교해 보세요."}
        <br />
        빠른 예제 설정은 연구용 수렴 기준이 아닙니다.
      </div>
      <button className="run-button" onClick={collect} disabled={busy}>
        <Play size={16} fill="currentColor" />
        {busy ? "작업을 등록하는 중…" : "계산 시작"}
        <span>RUN</span>
      </button>
    </aside>
  );
}

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
import type {
  Atom,
  Catalog,
  CatalogItem,
  Example,
  Method,
  Spec,
  Task,
} from "./types";
import { atomText, CORRELATED_METHODS, itemName, taskLabels } from "./types";

export default function Settings({
  spec,
  setSpec,
  examples,
  catalog,
  expert,
  onRun,
  onPrepareExample,
  busy,
  onError,
}: {
  spec: Spec;
  setSpec: (s: Spec) => void;
  examples: Example[];
  catalog: Catalog;
  expert: boolean;
  onRun: (s: Spec) => void;
  onPrepareExample: () => void;
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
      const enteredAtoms = coordinates
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
      // The textarea is rounded for readability. Preserve imported/optimized
      // coordinates at full precision until the user actually edits them.
      const atoms =
        coordinates === atomText(spec.atoms) ? spec.atoms : enteredAtoms;
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
    "tdhf",
    "frequencies",
    "eomccsd",
    "educational_rhf",
  ];
  const correlated = CORRELATED_METHODS.includes(spec.method);
  const activeSpace = ["CASCI", "CASSCF", "NEVPT2"].includes(spec.method);
  const closedShellTask =
    correlated ||
    [
      "tda",
      "tddft",
      "tdhf",
      "frequencies",
      "eomccsd",
      "educational_rhf",
    ].includes(spec.task);
  const methods: Method[] =
    spec.task === "eomccsd"
      ? ["CCSD"]
      : spec.domain === "crystal" || spec.task === "tddft"
        ? ["DFT"]
        : ["educational_rhf", "tdhf"].includes(spec.task)
          ? ["HF"]
          : spec.task === "single_point"
            ? ["HF", "DFT", ...CORRELATED_METHODS]
            : ["HF", "DFT"];
  function chooseTask(task: Task) {
    const method: Method =
      task === "eomccsd"
        ? "CCSD"
        : ["tdhf", "educational_rhf"].includes(task)
          ? "HF"
          : task === "tddft"
            ? "DFT"
            : correlated && task !== "single_point"
              ? "HF"
              : spec.method;
    update({
      task,
      method,
      frozen_orbitals: 0,
      ir_intensities: false,
      ...([
        "tda",
        "tddft",
        "tdhf",
        "frequencies",
        "eomccsd",
        "educational_rhf",
      ].includes(task)
        ? { spin: 0, restricted: true }
        : {}),
      ...(task === "frequencies" || task === "educational_rhf"
        ? { ecp: null }
        : {}),
    });
  }
  function chooseMethod(method: Method) {
    update({
      method,
      frozen_orbitals: 0,
      ...(CORRELATED_METHODS.includes(method)
        ? { spin: 0, restricted: true }
        : {}),
    });
  }
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
  const hydrogenExample = examples.find((example) => example.id === "h2");
  const runButton = (
    <button className="run-button" onClick={collect} disabled={busy}>
      <Play size={16} fill="currentColor" />
      {busy ? "작업을 등록하는 중…" : "계산 시작"}
    </button>
  );
  return (
    <aside className="settings-panel">
      <div className="panel-label">
        <Settings2 size={15} /> 계산 설정
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
      {!expert && (
        <section className="learner-start" aria-label="첫 계산 안내">
          <strong>처음이라면 수소 분자부터</strong>
          <p>원자 2개인 H₂로 계산과 그래프를 익힐 수 있습니다.</p>
          <button
            className="text-button"
            disabled={!hydrogenExample || busy}
            onClick={() => {
              if (hydrogenExample) {
                load(hydrogenExample.spec);
                onPrepareExample();
              }
            }}
          >
            H₂ 예제 불러오기
          </button>
          <ol>
            <li>예제를 불러옵니다. 설정은 그대로 두어도 됩니다.</li>
            <li>계산 시작을 누릅니다.</li>
            <li>수렴 그래프를 보고 전자밀도를 만들어 봅니다.</li>
          </ol>
          {runButton}
          <small>
            예제 불러오기는 입력만 준비하며 계산을 시작하지 않습니다.
          </small>
        </section>
      )}
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
          wrap="off"
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
              aria-describedby={!expert ? "charge-help" : undefined}
              value={spec.charge}
              onChange={(e) => update({ charge: Number(e.target.value) })}
            />
            {!expert && (
              <small id="charge-help">
                0은 중성입니다. +1은 전자가 하나 적고, −1은 하나 많습니다.
              </small>
            )}
          </label>
          <label className="field">
            스핀 <small>Nα − Nβ</small>
            <input
              type="number"
              min={0}
              aria-label="스핀"
              aria-describedby={!expert ? "spin-help" : undefined}
              disabled={closedShellTask}
              value={spec.spin}
              onChange={(e) =>
                update({
                  spin: Number(e.target.value),
                  restricted:
                    Number(e.target.value) === 0 ? spec.restricted : false,
                })
              }
            />
            {!expert && (
              <small id="spin-help">
                두 스핀 방향의 전자 수 차이입니다. 다중도 값이 아닙니다. H₂·물
                기본 예제는 0입니다.
              </small>
            )}
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
          onChange={(e) => chooseTask(e.target.value as Task)}
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
            disabled={methods.length === 1}
            onChange={(e) => chooseMethod(e.target.value as Method)}
          >
            {methods.map((method) => (
              <option key={method} value={method}>
                {method === "HF" ? "Hartree–Fock" : method}
              </option>
            ))}
          </select>
        </label>
        <label className="field">
          기저 집합
          <select
            aria-label="기저 집합"
            aria-describedby={!expert ? "basis-help" : undefined}
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
      {!expert && (
        <p className="setting-help" id="basis-help">
          기저는 전자 상태의 공간 모양을 표현하는 수학 함수들의 집합입니다. 큰
          기저는 표현을 풍부하게 하지만 계산량도 늘립니다. STO-3G는 원리를 보는
          작은 학습용 기저입니다.
        </p>
      )}
      {correlated && (
        <div className="method-settings">
          <p className="settings-explanation">
            닫힌껍질 RHF 기준의 {spec.method} 단일점 계산입니다. 전체 에너지는
            상관 방법의 결과이며, 표시하는 오비탈·전자밀도·쌍극자는 RHF 기준
            상태입니다.
          </p>
          {["MP2", "CCSD", "CCSD(T)", "CISD"].includes(spec.method) && (
            <label className="field">
              동결 점유 오비탈 수
              <input
                type="number"
                aria-label="동결 오비탈 수"
                min={0}
                step={1}
                value={spec.frozen_orbitals ?? 0}
                onChange={(e) =>
                  update({ frozen_orbitals: Number(e.target.value) })
                }
              />
              <small>
                낮은 에너지의 점유 공간 오비탈을 상관 계산에서 제외합니다. 0은
                전체 전자 상관입니다.
              </small>
            </label>
          )}
          {activeSpace && (
            <>
              <div className="field-row">
                <label className="field">
                  활성 전자 수
                  <input
                    type="number"
                    aria-label="활성 전자 수"
                    min={2}
                    step={2}
                    value={spec.active_electrons ?? 2}
                    onChange={(e) =>
                      update({ active_electrons: Number(e.target.value) })
                    }
                  />
                </label>
                <label className="field">
                  활성 오비탈 수
                  <input
                    type="number"
                    aria-label="활성 오비탈 수"
                    min={1}
                    max={spec.method === "NEVPT2" ? 8 : 12}
                    step={1}
                    value={spec.active_orbitals ?? 2}
                    onChange={(e) =>
                      update({ active_orbitals: Number(e.target.value) })
                    }
                  />
                </label>
              </div>
              <p className="settings-explanation">
                초기 활성 공간은 닫힌껍질 코어 다음의 연속된 정준 RHF
                오비탈입니다. CASSCF에서는 오비탈도 최적화합니다. 자동 선택이
                화학적으로 적절한 활성 공간을 보장하지 않습니다.
              </p>
            </>
          )}
          {spec.method === "NEVPT2" && (
            <p className="settings-explanation">
              정준 RHF 오비탈의 CASCI 기준에 strongly-contracted NEVPT2 보정을
              더합니다. CASSCF 오비탈 최적화를 자동 수행하지 않습니다.
            </p>
          )}
          {spec.task === "eomccsd" && (
            <p className="settings-explanation">
              CCSD 바닥상태에서 singlet EOM 여기 에너지와 잔차를 계산합니다.
              전이 세기·흡수 스펙트럼·NTO는 제공하지 않습니다.
            </p>
          )}
          {spec.method === "FCI" && (
            <p className="settings-explanation">
              FCI는 유한한 기저 안의 정확한 전자 풀이입니다. 기저·상대론·핵
              운동의 오차는 남으며, 큰 determinant 공간은 자원 보호를 위해 실행
              전에 거부합니다.
            </p>
          )}
        </div>
      )}
      <div className="catalog-status">
        기저: {supportText(basisInfo)}
        {basisInfo?.description && <small>{basisInfo.description}</small>}
      </div>
      {spec.method === "DFT" && (
        <label className="field">
          교환·상관 함수형
          <input
            aria-label="함수형"
            aria-describedby={!expert ? "functional-help" : undefined}
            list="functional-list"
            value={spec.functional}
            disabled={spec.domain === "crystal"}
            onChange={(e) => update({ functional: e.target.value })}
          />
          {!expert && (
            <small id="functional-help">
              DFT에서 전자 사이 상호작용의 일부를 근사하는 계산 규칙입니다. 모든
              문제에 가장 정확한 한 가지 함수형은 없습니다.
            </small>
          )}
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
      {["tda", "tddft", "tdhf", "eomccsd"].includes(spec.task) && (
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
      {spec.task === "frequencies" && (
        <div className="method-settings">
          <p className="settings-explanation">
            입력 구조에서 analytic Hessian을 계산합니다. 먼저 구조를 최적화하고,
            완료된 최적화 작업의 조건을 복사해 진동 분석을 실행하세요. 허수
            진동수는 음수로 표시합니다.
          </p>
          <label className="check-field">
            <input
              type="checkbox"
              checked={spec.ir_intensities ?? false}
              onChange={(e) => update({ ir_intensities: e.target.checked })}
            />
            IR 세기 계산 (변위 구조의 추가 SCF 필요)
          </label>
          <div className="field-row">
            <label className="field">
              온도 / K
              <input
                type="number"
                aria-label="열화학 온도"
                min={1}
                value={spec.temperature_kelvin ?? 298.15}
                onChange={(e) =>
                  update({ temperature_kelvin: Number(e.target.value) })
                }
              />
            </label>
            <label className="field">
              압력 / Pa
              <input
                type="number"
                aria-label="열화학 압력"
                min={1}
                value={spec.pressure_pa ?? 101325}
                onChange={(e) =>
                  update({ pressure_pa: Number(e.target.value) })
                }
              />
            </label>
          </div>
          <p className="settings-explanation">
            이상기체·강체 회전자·조화 진동 근사입니다. 비정상 구조 또는 비양의
            진동 모드가 있으면 열화학 값을 제공하지 않습니다.
          </p>
        </div>
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
                  disabled={closedShellTask}
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
      {expert && runButton}
    </aside>
  );
}

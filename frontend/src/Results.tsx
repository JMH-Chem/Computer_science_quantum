import { useMemo, useState } from "react";
import {
  ArrowDown,
  BookOpen,
  ChevronRight,
  Info,
  Layers,
  TrendingDown,
} from "lucide-react";
import type { Data } from "plotly.js";
import Chart from "./Chart";
import type { Catalog, Job, JobEvent, Result } from "./types";
import { taskLabels } from "./types";
import { comparisonProblem, finalEnergy } from "./jobComparison";

export function Convergence({
  events,
  onSelect,
  selected,
}: {
  events: JobEvent[];
  onSelect: (event: JobEvent) => void;
  selected: number | null;
}) {
  const iterations = useMemo(
    () => events.filter((e) => e.type === "scf_iteration"),
    [events],
  );
  const [metric, setMetric] = useState("energy_hartree");
  const chart = useMemo<Data[]>(
    () => [
      {
        x: iterations.map((_, i) => i + 1),
        y: iterations.map((e) =>
          typeof e.data[metric] === "number" && Number.isFinite(e.data[metric])
            ? (e.data[metric] as number)
            : null,
        ),
        type: "scatter",
        mode: "lines+markers",
        line: {
          color: metric === "energy_hartree" ? "#64dfc4" : "#b0a1fa",
          width: 2,
        },
        marker: {
          size: iterations.map((e) => (e.seq === selected ? 10 : 5)),
          color: iterations.map((e) =>
            e.seq === selected ? "#ffffff" : "#64dfc4",
          ),
        },
        customdata: iterations.map((e) => [
          Number(e.data.iteration),
          String(e.data.optimization_step ?? "-"),
        ]),
        hovertemplate:
          "기록 %{x} · SCF 반복 %{customdata[0]}<br>%{y:.10g}<extra></extra>",
      },
    ],
    [iterations, metric, selected],
  );
  const current =
    iterations.find((e) => e.seq === selected) ?? iterations.at(-1);
  return (
    <>
      <div className="panel-heading">
        <div>
          <span className="eyebrow">SELF-CONSISTENT FIELD</span>
          <h2>
            수렴을 따라가다{" "}
            <span className="count-pill">{iterations.length} STEPS</span>
          </h2>
        </div>
        <select
          aria-label="수렴 그래프 지표"
          value={metric}
          onChange={(e) => setMetric(e.target.value)}
        >
          <option value="energy_hartree">전체 에너지</option>
          <option value="gradient_norm">궤도 기울기</option>
          <option value="density_change">밀도 변화</option>
        </select>
      </div>
      {iterations.length ? (
        <>
          <Chart
            data={chart}
            xTitle="SCF 기록 순서 (시간 아님)"
            yTitle={
              metric === "energy_hartree"
                ? "Energy / Eh"
                : metric === "gradient_norm"
                  ? "Orbital gradient norm"
                  : "Density change norm"
            }
            height={260}
            layout={{
              yaxis: {
                type: metric === "energy_hartree" ? "linear" : "log",
                gridcolor: "#21313e",
                title: {
                  text:
                    metric === "energy_hartree" ? "Energy / Eh" : "Norm (log)",
                },
                automargin: true,
              },
            }}
            onPoint={(i) => onSelect(iterations[i])}
          />
          <div className="iteration-readout">
            <span>
              선택한 SCF 반복{" "}
              <strong>{String(current?.data.iteration ?? "—")}</strong>
            </span>
            <span>
              ΔE{" "}
              <strong>
                {scientific(current?.data.delta_energy)} <small>Eh</small>
              </strong>
            </span>
            <span>
              밀도 변화{" "}
              <strong>{scientific(current?.data.density_change)}</strong>
            </span>
          </div>
        </>
      ) : (
        <div className="empty-chart">
          <TrendingDown size={32} />
          <strong>첫 번째 계산을 기다리고 있습니다</strong>
          <p>계산을 시작하면 실제 반복값이 여기에 쌓입니다.</p>
          <div className="empty-grid" />
        </div>
      )}
      <p className="science-note">
        <Info size={14} />
        SCF는 전자 분포를 찾는 수치 반복입니다. 에너지는 오르내릴 수 있으며,
        가로축은 물리적 시간이 아닙니다.
      </p>
    </>
  );
}
function scientific(v: unknown) {
  return typeof v === "number" && Number.isFinite(v) ? v.toExponential(3) : "—";
}
export function Orbitals({ result }: { result: Result | null | undefined }) {
  const orbitals = result?.orbital_energies_hartree;
  const [spin, setSpin] = useState(0);
  if (!orbitals?.length)
    return (
      <Empty
        label="전자 구조 결과를 기다립니다"
        description="분자 계산이 수렴하면 궤도 에너지와 점유수가 표시됩니다."
      />
    );
  const unrestricted = Array.isArray(orbitals[0]);
  const values = (unrestricted ? orbitals[spin] : orbitals) as number[];
  const raw = result?.occupations;
  const occ = (Array.isArray(raw?.[0]) ? raw?.[spin] : raw) as
    number[] | undefined;
  return (
    <>
      <div className="panel-heading">
        <div>
          <span className="eyebrow">MOLECULAR ORBITALS</span>
          <h2>전자 구조</h2>
        </div>
        {unrestricted && (
          <select
            aria-label="궤도 스핀 채널"
            value={spin}
            onChange={(e) => setSpin(Number(e.target.value))}
          >
            <option value={0}>α 스핀</option>
            <option value={1}>β 스핀</option>
          </select>
        )}
      </div>
      <Chart
        data={[
          {
            x: values.map((_, i) => i + 1),
            y: values.map((e) => e * 27.211386245988),
            type: "bar",
            marker: {
              color: values.map((_, i) =>
                (occ?.[i] ?? 0) > 0 ? "#64dfc4" : "#9286ca",
              ),
            },
            customdata: values.map((_, i) => occ?.[i] ?? 0),
            hovertemplate:
              "MO %{x}<br>%{y:.5f} eV<br>점유수 %{customdata}<extra></extra>",
          },
        ]}
        xTitle="궤도 번호 (1부터)"
        yTitle="궤도 에너지 / eV"
      />
      <p className="science-note">
        <Info size={14} />
        궤도 에너지의 차이는 광학 여기 에너지와 같지 않습니다. 3D 색상은
        오비탈의 위상을 나타냅니다.
      </p>
    </>
  );
}
export function Excitations({ result }: { result: Result | null | undefined }) {
  const allStates = result?.excitations ?? [];
  const states = allStates.filter(
    (s) =>
      s.converged !== false &&
      Number.isFinite(s.energy_ev) &&
      s.energy_ev > 0 &&
      Number.isFinite(s.oscillator_strength) &&
      s.oscillator_strength >= 0,
  );
  const [width, setWidth] = useState(0.15),
    [broadening, setBroadening] = useState("gaussian"),
    [animate, setAnimate] = useState(false);
  const data = useMemo<Data[]>(() => {
    if (!states.length) return [];
    const min = Math.max(0, Math.min(...states.map((s) => s.energy_ev)) - 1),
      max = Math.max(...states.map((s) => s.energy_ev)) + 1;
    const x = Array.from(
      { length: 500 },
      (_, i) => min + ((max - min) * i) / 499,
    );
    const y = x.map((e) =>
      states.reduce(
        (sum, s) =>
          sum +
          s.oscillator_strength *
            (broadening === "gaussian"
              ? Math.exp(-0.5 * ((e - s.energy_ev) / width) ** 2) /
                (width * Math.sqrt(2 * Math.PI))
              : width / Math.PI / ((e - s.energy_ev) ** 2 + width ** 2)),
        0,
      ),
    );
    return [
      {
        x,
        y,
        type: "scatter",
        mode: "lines",
        name: broadening === "gaussian" ? "Gaussian" : "Lorentzian",
        line: { color: "#64dfc4" },
        fill: "tozeroy",
        fillcolor: "rgba(100,223,196,.09)",
        yaxis: "y2",
      },
      {
        x: states.flatMap((s) => [s.energy_ev, s.energy_ev, null]),
        y: states.flatMap((s) => [0, s.oscillator_strength, null]),
        type: "scatter",
        mode: "lines",
        name: "계산된 전이 세기",
        line: { color: "#bbabfc", width: 2 },
        hovertemplate: "%{x:.4f} eV<br>f=%{y:.6f}<extra></extra>",
      },
    ];
  }, [states, width, broadening]);
  if (!states.length)
    return (
      <Empty
        label="여기 상태를 탐색해 보세요"
        description={
          allStates.length
            ? "수렴하고 유효한 여기 상태가 없습니다. 계산 진단과 원본 결과를 확인하세요."
            : "TDA 또는 TDDFT 계산 후 수직 여기 에너지, 흡수 세기와 상태도를 확인할 수 있습니다."
        }
      />
    );
  const max = Math.max(...states.map((s) => s.energy_ev));
  return (
    <>
      {(!result?.converged || allStates.length !== states.length) && (
        <div className="warning-note">
          일부 여기 상태가 수렴하지 않았거나 유효하지 않습니다. 아래에는
          유효하고 수렴한 상태만 표시합니다. 전체 계산 결과는 연구용으로
          확정되지 않았습니다.
        </div>
      )}
      <div className="panel-heading">
        <div>
          <span className="eyebrow">VERTICAL EXCITATIONS</span>
          <h2>흡수 스펙트럼과 상태도</h2>
        </div>
        <span className="small-tag">{states.length} states</span>
      </div>
      <div className="spectrum-controls">
        <label>
          표시용 선폭
          <input
            aria-label="스펙트럼 선폭"
            type="range"
            min={0.03}
            max={0.5}
            step={0.01}
            value={width}
            onChange={(e) => setWidth(Number(e.target.value))}
          />
          <strong>{width.toFixed(2)} eV</strong>
        </label>
        <select
          aria-label="스펙트럼 퍼짐 함수"
          value={broadening}
          onChange={(e) => setBroadening(e.target.value)}
        >
          <option value="gaussian">Gaussian · σ</option>
          <option value="lorentzian">Lorentzian · HWHM</option>
        </select>
      </div>
      <Chart
        data={data}
        xTitle="여기 에너지 / eV"
        yTitle="Oscillator strength"
        layout={{
          showlegend: true,
          legend: { orientation: "h", y: 1.14, font: { size: 10 } },
          yaxis2: {
            overlaying: "y",
            side: "right",
            showgrid: false,
            title: { text: "Broadened / eV⁻¹" },
            rangemode: "tozero",
          },
        }}
      />
      <div className="state-layout">
        <div className={`state-diagram ${animate ? "animating" : ""}`}>
          <span className="state-axis">여기 에너지</span>
          <div className="state-line ground">
            <span>S₀</span>
            <b>0 eV</b>
          </div>
          {states.slice(0, 6).map((s) => (
            <div
              key={s.state}
              className="state-line excited"
              style={{ bottom: `${45 + (150 * s.energy_ev) / max}px` }}
            >
              <span>S{s.state}</span>
              <b>{s.energy_ev.toFixed(3)} eV</b>
            </div>
          ))}
          <div
            className="transition-path"
            style={{
              top: "auto",
              bottom: 45,
              height: (150 * states[0].energy_ev) / max,
            }}
          >
            <ArrowDown size={28} />
          </div>
          <span className="diagram-label">교육용 모식도 · 시간 궤적 아님</span>
        </div>
        <div className="state-description">
          <span className="eyebrow">UNDERSTAND THE TRANSITION</span>
          <h3>S₁에서 S₀로 돌아간다는 것</h3>
          <p>
            계산된 값은 현재 구조에서의 수직 여기 에너지입니다. 아래 방향
            화살표는 전이 개념을 설명하며, 실제 방출 에너지·수명·전이 확률을
            예측한 결과가 아닙니다.
          </p>
          <button
            className="secondary-button"
            onClick={() => setAnimate((a) => !a)}
          >
            {animate ? "설명 애니메이션 정지" : "전이 개념 보기"}
            <ChevronRight size={14} />
          </button>
        </div>
      </div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>상태</th>
              <th>여기 에너지 / eV</th>
              <th>Oscillator strength</th>
              <th>주요 NTO 가중치</th>
            </tr>
          </thead>
          <tbody>
            {states.map((s) => (
              <tr key={s.state}>
                <td>S{s.state}</td>
                <td>{s.energy_ev.toFixed(6)}</td>
                <td>{s.oscillator_strength.toFixed(6)}</td>
                <td>
                  {s.nto_weights
                    ?.slice(0, 3)
                    .map((n) => n.toFixed(4))
                    .join(" · ") ?? "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="science-note">
        <Info size={14} />
        퍼짐 함수와 선폭은 표시 설정입니다. 계산된 전이 에너지와 세기는 바뀌지
        않습니다. NTO 정의는 엔진의 X-only 관례를 따릅니다.
      </p>
    </>
  );
}
export function Bands({ result }: { result: Result | null | undefined }) {
  const bands = result?.bands;
  if (!bands)
    return (
      <Empty
        label="결정의 밴드를 확인하세요"
        description="재료의 밴드 구조 계산 후 고대칭 경로의 Kohn–Sham 에너지를 표시합니다."
      />
    );
  const data: Data[] = Array.from(
    { length: bands.energies_ev[0]?.length ?? 0 },
    (_, i) => ({
      x: bands.distance.flatMap((d, k) =>
        bands.break_indices?.includes(k) ? [null, d] : [d],
      ),
      y: bands.energies_ev.flatMap((row, k) =>
        bands.break_indices?.includes(k) ? [null, row[i]] : [row[i]],
      ),
      type: "scatter",
      mode: "lines",
      connectgaps: false,
      line: {
        color:
          i < (Number(bands["n_occupied" as keyof typeof bands]) || 4)
            ? "#64dfc4"
            : "#ab9bea",
        width: 1.6,
      },
      hovertemplate: "k경로 %{x:.3f}<br>%{y:.5f} eV<extra></extra>",
    }),
  );
  return (
    <>
      <div className="panel-heading">
        <div>
          <span className="eyebrow">PERIODIC ELECTRONIC STRUCTURE</span>
          <h2>밴드 구조</h2>
        </div>
        {typeof bands.gap_ev === "number" && (
          <span className="small-tag">
            표본 KS gap {bands.gap_ev.toFixed(3)} eV
          </span>
        )}
      </div>
      <Chart
        data={data}
        yTitle="E − VBM / eV"
        height={360}
        layout={{
          xaxis: {
            tickvals: bands.labels.map((l) => l.distance),
            ticktext: bands.labels.map((l) => l.label),
            gridcolor: "#3a4d5c",
            title: { text: "고대칭 k 경로" },
          },
          yaxis: {
            title: { text: "E − VBM / eV" },
            gridcolor: "#21313e",
            zerolinecolor: "#b0a1fa",
            zerolinewidth: 1.5,
          },
        }}
      />
      <p className="science-note">
        <Info size={14} />
        표본 경로에서 얻은 Kohn–Sham 밴드갭입니다. 광학 갭과 같지 않으며,
        기저·k점·수치 격자 수렴을 별도로 확인해야 합니다.
      </p>
    </>
  );
}
export function Compare({
  jobs,
  current,
}: {
  jobs: Job[];
  current: Job | null;
}) {
  const [referenceId, setReferenceId] = useState<string | null>(null);
  const [compatibleOnly, setCompatibleOnly] = useState(true);
  const valid = jobs.filter(
    (job) =>
      typeof job.result?.energy_hartree === "number" &&
      Number.isFinite(job.result.energy_hartree) &&
      job.spec.task !== "volume",
  );
  const reference =
    valid.find((job) => job.id === referenceId) ??
    (current && finalEnergy(current) !== null
      ? current
      : valid.find((job) => finalEnergy(job) !== null));
  const rows = valid
    .map((job) => ({
      job,
      problem: reference
        ? comparisonProblem(reference, job)
        : "수렴한 기준 계산 없음",
    }))
    .filter((row) => !compatibleOnly || row.problem === null);
  const perAtom = reference?.spec.domain === "crystal";
  const shownEnergy = (job: Job) =>
    typeof job.result?.energy_hartree === "number"
      ? job.result.energy_hartree / (perAtom ? job.spec.atoms.length : 1)
      : NaN;
  return (
    <div className="advanced-results">
      <div className="panel-heading">
        <div>
          <span className="eyebrow">REPRODUCIBLE COMPARISON</span>
          <h2>같은 구조·모델의 수치 조건 비교</h2>
        </div>
        <Layers size={20} />
      </div>
      {valid.length ? (
        <>
          <div className="comparison-controls">
            <label>
              기준 계산{" "}
              <select
                aria-label="비교 기준 계산"
                value={reference?.id ?? ""}
                onChange={(event) => setReferenceId(event.target.value)}
              >
                {valid
                  .filter((job) => finalEnergy(job) !== null)
                  .map((job) => (
                    <option value={job.id} key={job.id}>
                      {job.spec.name} · {job.id.slice(0, 6)}
                    </option>
                  ))}
              </select>
            </label>
            <label className="check-field">
              <input
                type="checkbox"
                checked={compatibleOnly}
                onChange={(event) => setCompatibleOnly(event.target.checked)}
              />
              비교 가능한 계산만 보기
            </label>
          </div>
          <div className="table-wrap">
            <table data-testid="job-comparison">
              <thead>
                <tr>
                  <th>계산</th>
                  <th>방법 · 함수형</th>
                  <th>기저</th>
                  <th>작업</th>
                  <th>전하 / 스핀</th>
                  <th>격자 / k점</th>
                  <th>전체 입력</th>
                  <th>에너지 / {perAtom ? "Eh atom⁻¹" : "Eh"}</th>
                  <th>기준과 ΔE / {perAtom ? "Eh atom⁻¹" : "Eh"}</th>
                  <th>비교 진단</th>
                </tr>
              </thead>
              <tbody>
                {rows.map(({ job, problem }) => (
                  <tr
                    key={job.id}
                    className={job.id === reference?.id ? "selected-row" : ""}
                  >
                    <td>{job.spec.name}</td>
                    <td>
                      {job.spec.method}
                      {job.spec.method === "DFT"
                        ? ` / ${job.spec.functional}`
                        : ""}
                    </td>
                    <td>
                      {typeof job.spec.basis === "string"
                        ? job.spec.basis
                        : "원소별 설정"}
                    </td>
                    <td>{taskLabels[job.spec.task]}</td>
                    <td>
                      {job.spec.charge} / {job.spec.spin}
                    </td>
                    <td>
                      {job.spec.domain === "crystal"
                        ? `${job.spec.mesh.join("×")} / ${job.spec.kmesh.join("×")}`
                        : `DFT grid ${job.spec.grid_level}`}
                    </td>
                    <td>
                      <a
                        href={`/api/jobs/${job.id}/export?format=json`}
                        target="_blank"
                        rel="noreferrer"
                      >
                        입력·결과
                      </a>
                    </td>
                    <td className="number-cell">
                      {shownEnergy(job).toFixed(9)}
                      {job.status !== "succeeded" && " (미수렴)"}
                    </td>
                    <td>
                      {!problem && reference
                        ? (shownEnergy(job) - shownEnergy(reference)).toFixed(9)
                        : "—"}
                    </td>
                    <td>
                      {problem ??
                        (job.id === reference?.id
                          ? "기준"
                          : "구조·전자 모델·버전 일치")}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {!rows.length && (
            <p>
              선택한 기준과 비교 가능한 계산이 없습니다. 전체 계산을 표시하면
              제외 사유를 확인할 수 있습니다.
            </p>
          )}
        </>
      ) : (
        <Empty
          label="비교할 계산이 아직 없습니다"
          description="같은 구조에서 기저나 수치 격자를 바꾸어 계산을 추가해 보세요."
        />
      )}
      <p className="science-note">
        <Info size={14} /> 최종 구조·전하·스핀·방법·함수형·핵심 전자 처리·엔진
        버전이 같은 수렴 결과만 차이를 계산합니다. 좌표 정렬이나 에너지 보정은
        하지 않습니다. 기저·수치 격자·k점·허용오차는 달라도 표에 함께 표시하므로
        한 조건씩 비교하세요. 다른 방법의 비교는 학습 실험의 방법 비교를
        사용하세요.
      </p>
    </div>
  );
}
export function Learn({
  catalog,
  result,
}: {
  catalog: Catalog;
  result: Result | null | undefined;
}) {
  const education = result?.education;
  return (
    <>
      <div className="panel-heading">
        <div>
          <span className="eyebrow">LEARN WHILE YOU COMPUTE</span>
          <h2>계산 뒤의 물리 이해하기</h2>
        </div>
        <BookOpen size={21} />
      </div>
      <div className="lesson-grid">
        {catalog.lessons.length ? (
          catalog.lessons.map((l, i) => (
            <article className="lesson-card" key={l.id ?? i}>
              <span className="lesson-number">
                {String(i + 1).padStart(2, "0")}
              </span>
              <h3>{l.title}</h3>
              <p>{l.body ?? l.content}</p>
              {(l.source ?? l.url) && (
                <a href={l.source ?? l.url} target="_blank" rel="noreferrer">
                  공식 자료 <ChevronRight size={12} />
                </a>
              )}
            </article>
          ))
        ) : (
          <article className="lesson-card">
            <h3>계산은 질문에서 시작됩니다</h3>
            <p>
              어떤 구조가 안정적인가요? 어떤 빛을 흡수하나요? 원하는 물리량에
              맞춰 계산 방법을 선택하고 수렴을 확인하세요.
            </p>
          </article>
        )}
      </div>
      {education && (
        <div className="education-result">
          <h3>자체 RHF 풀이 기록</h3>
          <p>
            적분은 PySCF가 제공하며, 직교화·Fock 행렬·밀도 갱신·SCF 반복은 자체
            NumPy 풀이기가 수행합니다.
          </p>
          {Object.entries(education).map(([key, value]) => (
            <details key={key}>
              <summary>{key}</summary>
              <pre>
                {typeof value === "string"
                  ? value
                  : JSON.stringify(value, null, 2)}
              </pre>
            </details>
          ))}
        </div>
      )}
    </>
  );
}
export function Empty({
  label,
  description,
}: {
  label: string;
  description: string;
}) {
  return (
    <div className="empty-result">
      <BookOpen size={28} />
      <h3>{label}</h3>
      <p>{description}</p>
    </div>
  );
}

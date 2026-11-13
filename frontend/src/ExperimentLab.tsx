import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowDown,
  ArrowDownToLine,
  ArrowRight,
  ArrowUp,
  Check,
  CircleHelp,
  History,
  Info,
  Layers3,
  LoaderCircle,
  LockKeyhole,
  Pause,
  Play,
  Plus,
  RefreshCw,
  Square,
} from "lucide-react";
import type { Data } from "plotly.js";
import Chart from "./Chart";
import Viewer from "./Viewer";
import { api } from "./api";
import { statusLabels, TERMINAL } from "./types";
import type {
  Experiment,
  ExperimentIntent,
  ExperimentRow,
  ExperimentTemplate,
  ExperimentVariable,
  Prediction,
} from "./experimentTypes";
import { EXPERIMENT_TERMINAL } from "./experimentTypes";
import "./experiment.css";

const SELECTED_KEY = "quantum-lab.experiment";
const PENDING_KEY = "quantum-lab.experiment.pending";
const TEMPLATE_KEY = "quantum-lab.experiment.template";
const predictionDefaults: {
  id: Prediction;
  label: string;
  icon: typeof ArrowDown;
}[] = [
  { id: "lower", label: "낮아질 것 같아요", icon: ArrowDown },
  { id: "higher", label: "높아질 것 같아요", icon: ArrowUp },
  { id: "unsure", label: "아직 모르겠어요", icon: CircleHelp },
];
const variableLabels: Record<ExperimentVariable, string> = {
  basis: "기저",
  grid_level: "적분 격자",
  functional: "함수형",
  bond_distance: "결합 거리",
  kmesh: "k점 격자",
  mesh: "FFT 격자",
  method: "계산 방법",
};
const basisDescriptions: Record<string, string> = {
  "sto-3g": "최소 기저 · 원리를 살펴보는 출발점",
  "def2-svp": "이중 제타 + 편극 함수",
  "def2-tzvp": "삼중 제타 + 편극 함수",
};
type Metric =
  "energy" | "delta" | "energy_per_atom" | "delta_per_atom" | "time" | "memory";
const metricLabels: Record<
  Metric,
  { label: string; axis: string; unit: string }
> = {
  energy: { label: "전체 에너지", axis: "전체 에너지 / Eh", unit: "Eh" },
  delta: {
    label: "이전 조건과의 차이",
    axis: "E현재 − E이전 / Eh",
    unit: "Eh",
  },
  energy_per_atom: {
    label: "원자당 에너지",
    axis: "에너지 / Eh atom⁻¹",
    unit: "Eh/atom",
  },
  delta_per_atom: {
    label: "원자당 에너지 차이",
    axis: "ΔE / Eh atom⁻¹",
    unit: "Eh/atom",
  },
  time: { label: "계산 시간", axis: "계산 프로세스 시간 / s", unit: "s" },
  memory: {
    label: "메모리 사용량",
    axis: "프로세스 peak RSS / MiB",
    unit: "MiB",
  },
};
function readPending(): ExperimentIntent | null {
  try {
    const value = JSON.parse(localStorage.getItem(PENDING_KEY) ?? "null");
    if (
      value &&
      typeof value.template_id === "string" &&
      typeof value.request_id === "string" &&
      typeof value.name === "string" &&
      predictionDefaults.some((item) => item.id === value.prediction)
    )
      return value;
  } catch {
    /* Invalid browser preferences do not become scientific inputs. */
  }
  return null;
}
function finite(value: number | null | undefined): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}
function fixed(value: number | null | undefined, digits = 3): string {
  const number = finite(value);
  return number === null ? "—" : number.toFixed(digits);
}
function rowLabel(row: ExperimentRow): string {
  return !row.label || row.label.toLowerCase() === row.basis.toLowerCase()
    ? row.basis.toUpperCase()
    : row.label;
}
function statusText(experiment: Experiment): string {
  if (experiment.status === "completed")
    return `${experiment.total_jobs}개 계산 완료`;
  if (experiment.status === "completed_with_issues")
    return "일부 계산에 진단 필요";
  if (experiment.status === "cancelled") return "실험 취소됨";
  if (experiment.status === "queued") return "실험 대기 중";
  return `${experiment.completed_jobs} / ${experiment.total_jobs}개 완료 · 계산 중`;
}
function initialName(template?: ExperimentTemplate | null): string {
  return template?.id === "water_basis_v1" || !template
    ? "물의 기저 비교"
    : template.title.replace(/[?？]$/, "");
}
function heading(templateId: string, title?: string): string {
  return templateId === "water_basis_v1"
    ? "물 분자의 기저를 바꾸면?"
    : (title ?? "계산 조건 비교 실험");
}

export default function ExperimentLab({
  onOpenJob,
}: {
  onOpenJob: (id: string) => void;
}) {
  const [templates, setTemplates] = useState<ExperimentTemplate[]>([]);
  const [draftTemplateId, setDraftTemplateId] = useState(
    () =>
      readPending()?.template_id ??
      localStorage.getItem(TEMPLATE_KEY) ??
      "water_basis_v1",
  );
  const [history, setHistory] = useState<Experiment[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(() =>
    localStorage.getItem(SELECTED_KEY),
  );
  const [experiment, setExperiment] = useState<Experiment | null>(null);
  const [pending, setPending] = useState<ExperimentIntent | null>(readPending);
  const [name, setName] = useState(
    () => readPending()?.name ?? "물의 기저 비교",
  );
  const [prediction, setPrediction] = useState<Prediction>(
    () => readPending()?.prediction ?? "unsure",
  );
  const [busy, setBusy] = useState(false),
    [cancelling, setCancelling] = useState(false);
  const [loading, setLoading] = useState(!!selectedId),
    [error, setError] = useState("");
  const [reload, setReload] = useState(0),
    [metric, setMetric] = useState<Metric>("energy");
  const [highlight, setHighlight] = useState<string | null>(null),
    [playing, setPlaying] = useState(false);
  const mounted = useRef(true),
    createInFlight = useRef(false);
  const template =
    templates.find(
      (item) => item.id === (pending?.template_id ?? draftTemplateId),
    ) ?? null;
  const currentTemplate = experiment
    ? templates.find((item) => item.id === experiment.template_id)
    : template;
  // Saved definitions are authoritative for historical experiments, including their explanations.
  const resolvingHistory = !!selectedId && !experiment;
  const presentation = resolvingHistory
    ? undefined
    : (experiment?.definition ?? currentTemplate);
  const currentTemplateId =
    experiment?.template_id ??
    history.find((record) => record.id === selectedId)?.template_id ??
    template?.id ??
    draftTemplateId;
  const variable =
    experiment?.definition?.variable ??
    experiment?.variable ??
    presentation?.variable ??
    "basis";
  const variableLabel =
    experiment?.definition?.variable_label ??
    experiment?.variable_label ??
    presentation?.variable_label ??
    variableLabels[variable];
  const baseSpec = resolvingHistory
    ? undefined
    : (experiment?.base_spec ?? template?.fixed_spec);
  const crystal = baseSpec?.domain === "crystal";
  const rawLabels = resolvingHistory
    ? []
    : (experiment?.definition?.labels ??
      experiment?.labels ??
      presentation?.labels ??
      experiment?.bases ??
      template?.bases ??
      []);
  const labels =
    variable === "basis"
      ? rawLabels.map((label) => label.toUpperCase())
      : rawLabels;
  const rows = experiment?.rows ?? [];
  const count = experiment?.total_jobs ?? labels.length;
  const sequenceKey = rows.map((row) => row.job_id).join(",");
  const running = !!experiment && !EXPERIMENT_TERMINAL.has(experiment.status);
  const activeIndex = Math.max(
    0,
    rows.findIndex((row) => row.job_id === highlight),
  );
  const displayedRow = rows[activeIndex];
  const displayedJob = experiment?.jobs.find(
    (job) => job.id === displayedRow?.job_id,
  );
  const displayedSpec =
    displayedJob?.spec ??
    experiment?.definition?.expected_specs?.[activeIndex] ??
    baseSpec;
  const displayAtoms =
    displayedJob?.status === "succeeded" && displayedJob.result?.atoms
      ? displayedJob.result.atoms
      : displayedSpec?.atoms;
  const predictions = predictionDefaults.map((item) => ({
    ...item,
    label:
      currentTemplateId === "water_basis_v1"
        ? item.label
        : (presentation?.predictions?.find((entry) => entry.id === item.id)
            ?.label ?? item.label),
  }));

  const refreshHistory = useCallback(async () => {
    const answer = await api<Experiment[]>("/experiments");
    if (mounted.current) setHistory(answer);
  }, []);
  useEffect(() => {
    mounted.current = true;
    let alive = true;
    void Promise.all([
      api<ExperimentTemplate[]>("/experiments/templates"),
      api<Experiment[]>("/experiments"),
    ])
      .then(([available, records]) => {
        if (!alive) return;
        if (!available.length)
          throw new Error("학습 실험을 불러오지 못했습니다.");
        setTemplates(available);
        setHistory(records);
        setDraftTemplateId((previous) =>
          available.some((item) => item.id === previous)
            ? previous
            : available[0].id,
        );
      })
      .catch((reason) => {
        if (alive) setError(String(reason));
      });
    return () => {
      alive = false;
      mounted.current = false;
    };
  }, [reload]);
  useEffect(() => {
    localStorage.setItem(TEMPLATE_KEY, draftTemplateId);
  }, [draftTemplateId]);
  useEffect(() => {
    setExperiment(null);
    setHighlight(null);
    setPlaying(false);
    if (!selectedId) {
      localStorage.removeItem(SELECTED_KEY);
      setLoading(false);
      return;
    }
    localStorage.setItem(SELECTED_KEY, selectedId);
    let alive = true;
    let timer: ReturnType<typeof setTimeout> | undefined;
    setLoading(true);
    const load = async () => {
      try {
        const answer = await api<Experiment>(`/experiments/${selectedId}`);
        if (!alive) return;
        setExperiment((previous) =>
          JSON.stringify(previous) === JSON.stringify(answer)
            ? previous
            : answer,
        );
        setLoading(false);
        setError("");
        if (EXPERIMENT_TERMINAL.has(answer.status))
          void refreshHistory().catch(() => {});
        else timer = setTimeout(load, 1000);
      } catch (reason) {
        if (!alive) return;
        setError(`실험 상태를 확인하지 못했습니다. ${String(reason)}`);
        setLoading(false);
        timer = setTimeout(load, 3000);
      }
    };
    void load();
    return () => {
      alive = false;
      if (timer) clearTimeout(timer);
    };
  }, [selectedId, reload, refreshHistory]);
  useEffect(() => {
    if (!playing || variable !== "bond_distance" || !sequenceKey.includes(","))
      return;
    const ids = sequenceKey.split(",");
    const timer = setInterval(
      () =>
        setHighlight(
          (previous) => ids[(ids.indexOf(previous ?? "") + 1) % ids.length],
        ),
      1200,
    );
    return () => clearInterval(timer);
  }, [playing, variable, sequenceKey]);

  async function start() {
    if ((!template && !pending) || createInFlight.current) return;
    const trimmedName = name.trim();
    if (!pending && !trimmedName) {
      setError("실험 이름을 입력하세요.");
      return;
    }
    const intent = pending ?? {
      template_id: template!.id,
      name: trimmedName,
      prediction,
      request_id: crypto.randomUUID(),
    };
    // Persist before network I/O. Retrying a lost response reuses the exact same request.
    localStorage.setItem(PENDING_KEY, JSON.stringify(intent));
    setPending(intent);
    createInFlight.current = true;
    setBusy(true);
    setError("");
    try {
      const response = await fetch("/api/experiments", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(intent),
      });
      if (!response.ok) {
        const body = await response.json().catch(() => null);
        if (response.status >= 400 && response.status < 500) {
          localStorage.removeItem(PENDING_KEY);
          if (mounted.current) setPending(null);
        }
        throw new Error(
          typeof body?.detail === "string"
            ? body.detail
            : `${response.status}: ${JSON.stringify(body?.detail ?? "실험을 생성하지 못했습니다.")}`,
        );
      }
      const created: Experiment = await response.json();
      localStorage.removeItem(PENDING_KEY);
      localStorage.setItem(SELECTED_KEY, created.id);
      if (!mounted.current) return;
      setPending(null);
      setSelectedId(created.id);
      setExperiment(created);
      setMetric("energy");
      void refreshHistory().catch(() => {});
    } catch (reason) {
      if (mounted.current) setError(String(reason));
    } finally {
      createInFlight.current = false;
      if (mounted.current) setBusy(false);
    }
  }
  async function cancel() {
    if (!experiment || cancelling) return;
    const id = experiment.id;
    setCancelling(true);
    try {
      const answer = await api<Experiment>(`/experiments/${id}/cancel`, {
        method: "POST",
      });
      if (mounted.current) {
        setExperiment((previous) => (previous?.id === id ? answer : previous));
        void refreshHistory().catch(() => {});
      }
    } catch (reason) {
      if (mounted.current) setError(String(reason));
    } finally {
      if (mounted.current) setCancelling(false);
    }
  }
  function prepareNew(nextTemplateId = currentTemplateId) {
    if (pending || busy) return;
    const next =
      templates.find((item) => item.id === nextTemplateId) ?? template;
    if (next) setDraftTemplateId(next.id);
    setSelectedId(null);
    setExperiment(null);
    setHighlight(null);
    setPlaying(false);
    setName(initialName(next));
    setPrediction("unsure");
    setError("");
    setMetric("energy");
  }
  const data = useMemo<Data[]>(() => {
    const values = rows.map((row) => {
      if (metric === "time") return finite(row.elapsed_seconds);
      if (metric === "memory") return finite(row.peak_memory_mb);
      if (!row.comparable) return null;
      if (metric === "energy_per_atom")
        return finite(row.energy_per_atom_hartree);
      if (metric === "delta_per_atom")
        return finite(row.delta_previous_per_atom_hartree);
      return finite(
        metric === "energy" ? row.energy_hartree : row.delta_previous_hartree,
      );
    });
    return [
      {
        x: rows.map((row) =>
          variable === "bond_distance" && typeof row.value === "number"
            ? row.value
            : rowLabel(row),
        ),
        y: values,
        type: "scatter",
        mode: "lines+markers",
        connectgaps: false,
        line: {
          color:
            metric === "time" || metric === "memory" ? "#ad6546" : "#285f8f",
          width: 2,
        },
        marker: {
          size: rows.map((row) => (row.job_id === highlight ? 12 : 8)),
        },
        hovertemplate: `%{x}<br>%{y:.8g} ${metricLabels[metric].unit}<extra></extra>`,
      },
    ];
  }, [rows, metric, highlight, variable]);
  const selectRow = useCallback(
    (index: number) => {
      if (rows[index]) {
        setPlaying(false);
        setHighlight(rows[index].job_id);
      }
    },
    [rows],
  );
  const chartAxis =
    variable === "bond_distance"
      ? "H–H 결합 거리 / Å"
      : variable === "basis"
        ? "기저 함수 집합"
        : variableLabel;
  const currentEnergyLabel = crystal
    ? "셀 전체 에너지 / Eh"
    : metricLabels.energy.axis;
  const selectedTitle = heading(currentTemplateId, presentation?.title);

  return (
    <main className="experiment-lab">
      <div className="lab-heading">
        <div>
          <h1>
            {selectedId && !experiment
              ? "저장된 실험을 불러오는 중"
              : selectedTitle}
          </h1>
          <p>조건별 계산값과 수렴 상태를 비교합니다.</p>
        </div>
        <button
          className="secondary-button"
          onClick={() => prepareNew()}
          disabled={busy || !!pending}
        >
          <Plus size={15} />새 실험 준비
        </button>
      </div>
      <section
        className="panel lab-template-picker"
        aria-label="실험 주제 선택"
      >
        <label className="field">
          학습 실험 주제
          <select
            aria-label="학습 실험 주제"
            value={currentTemplateId}
            disabled={busy || !!pending}
            onChange={(event) => prepareNew(event.target.value)}
          >
            {templates.map((item) => (
              <option key={item.id} value={item.id}>
                {item.domain === "crystal" ? "재료" : "분자"} ·{" "}
                {heading(item.id, item.title)}
              </option>
            ))}
            {experiment &&
              !templates.some((item) => item.id === experiment.template_id) && (
                <option value={experiment.template_id}>
                  {presentation?.title ?? experiment.template_id} · 저장된 정의
                </option>
              )}
          </select>
        </label>
        <div className="lab-template-description">
          <span>
            {crystal ? "주기계 · 결정" : "분자"} · {count}개 조건
          </span>
          <p>{presentation?.description ?? "선택한 계산 조건을 비교합니다."}</p>
          {experiment && (
            <small>
              이 화면의 조건과 해설은 실험 생성 시 저장한 정의를 사용합니다.
            </small>
          )}
        </div>
        <details className="lab-topic-list">
          <summary>실험 주제 둘러보기 · {templates.length}개</summary>
          <div>
            {templates.map((item) => (
              <button
                type="button"
                key={item.id}
                className={item.id === currentTemplateId ? "selected" : ""}
                disabled={busy || !!pending}
                onClick={() => prepareNew(item.id)}
              >
                <span>
                  {item.domain === "crystal" ? "재료" : "분자"} ·{" "}
                  {item.variable_label ?? variableLabels[item.variable]}
                </span>
                <strong>{heading(item.id, item.title)}</strong>
                <small>{item.description}</small>
              </button>
            ))}
          </div>
        </details>
      </section>
      <ol className="lab-steps" aria-label="실험 진행 단계">
        {[
          ["01", "예측하기", "내 생각부터 기록"],
          ["02", "계산하기", `${count}개 조건 · 순서대로`],
          ["03", "관찰하기", "에너지와 비용 비교"],
          ["04", "해석하기", "수렴의 의미 이해"],
        ].map(([number, title, detail], index) => (
          <li
            key={number}
            className={
              (!experiment && index === 0) ||
              (running && index === 1) ||
              (experiment && !running && index >= 2)
                ? "active"
                : ""
            }
          >
            <span>{number}</span>
            <div>
              <strong>{title}</strong>
              <small>{detail}</small>
            </div>
            {index < 3 && <ArrowRight size={15} />}
          </li>
        ))}
      </ol>
      {error && (
        <div className="lab-error" role="alert">
          <CircleHelp size={17} />
          <span>{error}</span>
          <button
            className="text-button"
            onClick={() => setReload((n) => n + 1)}
          >
            상태 다시 확인
          </button>
        </div>
      )}
      {pending && !busy && (
        <div className="lab-pending" role="status">
          <Info size={18} />
          <div>
            <strong>전송한 실험 요청을 확인해야 합니다.</strong>
            <p>
              응답이 끊겨도 서버에서 계산이 시작되었을 수 있습니다. 같은 요청을
              다시 보내면 기존 실험을 확인합니다.
            </p>
          </div>
          <button className="secondary-button" onClick={() => void start()}>
            <RefreshCw size={14} />
            같은 요청 다시 보내기
          </button>
        </div>
      )}
      <div className="lab-layout">
        <aside className="lab-sidebar">
          <section className="panel lab-conditions">
            <div className="panel-heading">
              <div>
                <h2>
                  <LockKeyhole size={16} />
                  고정하는 조건
                </h2>
              </div>
            </div>
            {baseSpec && displayedSpec && displayAtoms ? (
              <>
                <div className="lab-structure">
                  <Viewer
                    atoms={displayAtoms}
                    lattice={displayedSpec.lattice}
                  />
                  {variable === "bond_distance" && rows.length > 0 && (
                    <div className="lab-scan-controls">
                      <label className="field">
                        표시할 계산 구조
                        <select
                          aria-label="표시할 계산 구조"
                          value={displayedRow?.job_id ?? ""}
                          onChange={(event) => {
                            setPlaying(false);
                            setHighlight(event.target.value);
                          }}
                        >
                          {rows.map((row) => (
                            <option key={row.job_id} value={row.job_id}>
                              {rowLabel(row)} ·{" "}
                              {statusLabels[row.status] ?? row.status}
                            </option>
                          ))}
                        </select>
                      </label>
                      <button
                        className="secondary-button"
                        disabled={rows.length < 2}
                        onClick={() => setPlaying((previous) => !previous)}
                      >
                        {playing ? <Pause size={14} /> : <Play size={14} />}
                        {playing ? "구조 순서 보기 멈춤" : "구조 순서대로 보기"}
                      </button>
                      <p data-testid="scan-structure-caption">
                        {rowLabel(displayedRow)} ·{" "}
                        {displayedRow?.status === "succeeded"
                          ? "계산에 사용한 구조"
                          : "입력 구조"}
                        . 각 조건을 순서대로 표시하며 실제 시간 궤적이 아닙니다.
                      </p>
                    </div>
                  )}
                </div>
                <dl>
                  <div>
                    <dt>{crystal ? "결정 구조" : "분자 구조"}</dt>
                    <dd>
                      {baseSpec.atoms.length}개 원자 ·{" "}
                      {variable === "bond_distance"
                        ? "결합 거리 비교"
                        : "동일 좌표"}
                    </dd>
                  </div>
                  {variable !== "method" && (
                    <div>
                      <dt>계산 방법</dt>
                      <dd>
                        {baseSpec.method}
                        {baseSpec.method === "DFT" && variable !== "functional"
                          ? ` / ${baseSpec.functional}`
                          : ""}
                      </dd>
                    </div>
                  )}
                  {variable !== "basis" && (
                    <div>
                      <dt>기저 집합</dt>
                      <dd>
                        {typeof baseSpec.basis === "string"
                          ? baseSpec.basis
                          : "원소별 지정"}
                      </dd>
                    </div>
                  )}
                  <div>
                    <dt>전하 / 스핀</dt>
                    <dd>
                      {baseSpec.charge} / {baseSpec.spin}
                    </dd>
                  </div>
                  {!crystal &&
                    variable !== "grid_level" &&
                    baseSpec.method === "DFT" && (
                      <div>
                        <dt>적분 격자</dt>
                        <dd>Level {baseSpec.grid_level}</dd>
                      </div>
                    )}
                  {crystal && (
                    <>
                      <div>
                        <dt>가상퍼텐셜</dt>
                        <dd>{baseSpec.pseudo}</dd>
                      </div>
                      {variable !== "kmesh" && (
                        <div>
                          <dt>k점 격자</dt>
                          <dd>{baseSpec.kmesh.join(" × ")}</dd>
                        </div>
                      )}
                      {variable !== "mesh" && (
                        <div>
                          <dt>FFT 격자</dt>
                          <dd>{baseSpec.mesh.join(" × ")}</dd>
                        </div>
                      )}
                    </>
                  )}
                  <div>
                    <dt>SCF 허용오차</dt>
                    <dd>{baseSpec.conv_tol.toExponential(0)} Eh</dd>
                  </div>
                  <div>
                    <dt>CPU / 메모리 예산</dt>
                    <dd>
                      {baseSpec.threads} 스레드 / {baseSpec.memory_mb} MB
                    </dd>
                  </div>
                </dl>
                <details className="lab-coordinate-details">
                  <summary>
                    {variable === "bond_distance"
                      ? "표시한 구조의 좌표"
                      : "고정된 원자 좌표 확인"}{" "}
                    · Å
                  </summary>
                  <table>
                    <thead>
                      <tr>
                        <th>원소</th>
                        <th>x</th>
                        <th>y</th>
                        <th>z</th>
                      </tr>
                    </thead>
                    <tbody>
                      {displayAtoms.map((atom, i) => (
                        <tr key={i}>
                          <th>{atom.symbol}</th>
                          {atom.position.map((value, j) => (
                            <td key={j}>{value.toFixed(3)}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  {displayedSpec.lattice && (
                    <>
                      <p>격자 벡터 · Å</p>
                      <pre>
                        {displayedSpec.lattice
                          .map((vector) =>
                            vector.map((value) => value.toFixed(4)).join("  "),
                          )
                          .join("\n")}
                      </pre>
                    </>
                  )}
                </details>
              </>
            ) : (
              <p className="lab-loading">
                <LoaderCircle size={16} className="spin" />
                실험 조건을 불러오는 중…
              </p>
            )}
            <div className="lab-variable">
              <div className="panel-label">
                <Layers3 size={14} />
                바꾸는 조건: {variableLabel}
              </div>
              {labels.map((label, i) => (
                <div className="lab-basis-item" key={`${label}-${i}`}>
                  <span>{i + 1}</span>
                  <div>
                    <strong>{label}</strong>
                    {variable === "basis" && (
                      <small>
                        {basisDescriptions[label.toLowerCase()] ??
                          "선택한 기저 함수 집합"}
                      </small>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </section>
          <section className="panel lab-history">
            <div className="panel-label">
              <History size={15} />
              저장된 실험
            </div>
            <select
              aria-label="저장된 실험"
              value={selectedId ?? ""}
              disabled={busy || !!pending}
              onChange={(event) => {
                setError("");
                setExperiment(null);
                setHighlight(null);
                setPlaying(false);
                setMetric("energy");
                setLoading(!!event.target.value);
                setSelectedId(event.target.value || null);
              }}
            >
              <option value="">새 실험 준비 화면</option>
              {history.map((record) => (
                <option value={record.id} key={record.id}>
                  {record.name} ·{" "}
                  {new Date(record.created_at).toLocaleString("ko-KR")}
                </option>
              ))}
            </select>
            <p>
              실험과 개별 계산은 로컬에 보관됩니다. 다시 열어 이어서 확인할 수
              있습니다.
            </p>
          </section>
        </aside>
        <div className="lab-main">
          {!selectedId ? (
            <section className="panel lab-prediction">
              <h2>
                {presentation?.question ??
                  `${variableLabel}를 바꾸면 에너지는 어떻게 달라질까요?`}
              </h2>
              <p>
                예상한 방향을 기록한 뒤 계산값과 비교하세요. 변화가 한
                방향이라는 보장은 없습니다.
              </p>
              <div className="lab-prediction-options">
                {predictions.map(({ id, label, icon: Icon }) => (
                  <button
                    key={id}
                    className={prediction === id ? "selected" : ""}
                    aria-pressed={prediction === id}
                    disabled={busy || !!pending}
                    onClick={() => setPrediction(id)}
                  >
                    <Icon size={20} />
                    <span>{label}</span>
                    {prediction === id && <Check size={14} />}
                  </button>
                ))}
              </div>
              <label className="field">
                실험 이름
                <input
                  aria-label="실험 이름"
                  value={name}
                  maxLength={80}
                  disabled={busy || !!pending}
                  onChange={(event) => setName(event.target.value)}
                />
              </label>
              <div className="lab-start-line">
                <div>
                  <strong>한 가지 조건만 바꾸기</strong>
                  <small>
                    {variableLabel}만 바꾼 계산 {count}개를 하나의 대기열에서
                    실행합니다.
                  </small>
                  {crystal && (
                    <small>
                      k점과 FFT 격자를 늘리면 계산 시간이 길어질 수 있습니다.
                    </small>
                  )}
                </div>
                <button
                  className="run-button"
                  disabled={!template || busy || !!pending}
                  onClick={() => void start()}
                >
                  {busy ? (
                    <LoaderCircle size={16} className="spin" />
                  ) : (
                    <Play size={16} fill="currentColor" />
                  )}
                  {variable === "basis"
                    ? `기저 ${count}개 비교 시작`
                    : `${count}개 조건 비교 시작`}
                </button>
              </div>
            </section>
          ) : (
            <section className="panel lab-run-summary">
              <div className="lab-run-title">
                <div>
                  <h2>{experiment?.name ?? "저장된 실험을 불러오는 중"}</h2>
                </div>
                {loading ? (
                  <LoaderCircle className="spin" size={18} />
                ) : (
                  experiment && (
                    <span
                      className={`lab-status ${experiment.status}`}
                      data-testid="experiment-status"
                    >
                      {running ? (
                        <LoaderCircle className="spin" size={15} />
                      ) : experiment.status === "completed" ? (
                        <Check size={16} />
                      ) : (
                        <Info size={16} />
                      )}
                      {statusText(experiment)}
                    </span>
                  )
                )}
              </div>
              {experiment && (
                <>
                  <div
                    className="lab-progress"
                    role="progressbar"
                    aria-label="실험 계산 진행"
                    aria-valuemin={0}
                    aria-valuemax={experiment.total_jobs}
                    aria-valuenow={experiment.completed_jobs}
                  >
                    <span
                      style={{
                        width: `${(100 * experiment.completed_jobs) / Math.max(experiment.total_jobs, 1)}%`,
                      }}
                    />
                  </div>
                  <div className="lab-run-meta">
                    <span>
                      나의 예상:{" "}
                      <strong>
                        {predictions.find(
                          (item) => item.id === experiment.prediction,
                        )?.label ?? experiment.prediction}
                      </strong>
                    </span>
                    <span>
                      계산 수렴 {experiment.successful_jobs} /{" "}
                      {experiment.total_jobs}
                    </span>
                    {running && (
                      <button
                        className="stop-button"
                        onClick={() => void cancel()}
                        disabled={cancelling}
                      >
                        {cancelling ? (
                          <LoaderCircle size={12} className="spin" />
                        ) : (
                          <Square size={11} fill="currentColor" />
                        )}
                        실험 취소
                      </button>
                    )}
                  </div>
                  <p className="lab-run-note">
                    {running
                      ? "같은 계산 큐를 사용합니다. 다른 작업이 실행 중이면 차례를 기다립니다."
                      : "계산 완료는 실행 상태를 뜻합니다. 수치 조건의 충분함이나 실험값과의 정확도를 인증하지 않습니다."}
                  </p>
                </>
              )}
            </section>
          )}
          <section className="panel lab-observations">
            <div className="panel-heading">
              <div>
                <h2>계산 결과 비교</h2>
              </div>
              <select
                aria-label="실험 그래프 지표"
                value={metric}
                onChange={(event) => setMetric(event.target.value as Metric)}
              >
                {Object.entries(metricLabels)
                  .filter(([key]) => crystal || !key.includes("per_atom"))
                  .map(([key, value]) => (
                    <option key={key} value={key}>
                      {crystal && key === "energy"
                        ? "셀 전체 에너지"
                        : variable === "basis" && key === "delta"
                          ? "이전 기저와의 차이"
                          : value.label}
                    </option>
                  ))}
              </select>
            </div>
            {rows.length > 0 ? (
              <>
                <Chart
                  data={data}
                  xTitle={chartAxis}
                  yTitle={
                    metric === "energy"
                      ? currentEnergyLabel
                      : metricLabels[metric].axis
                  }
                  height={270}
                  onPoint={selectRow}
                />
                <p className="lab-chart-note">
                  {metric === "delta" || metric === "delta_per_atom"
                    ? "현재 에너지 − 바로 이전 조건의 에너지입니다. 첫 조건과 비교할 수 없는 항목은 비워 둡니다."
                    : metric === "time" || metric === "memory"
                      ? "이 실행에서 측정한 프로세스 비용입니다. 하드웨어와 실행 상황에 따라 달라지며, 메모리 예산과 실제 peak RSS는 다릅니다."
                      : "수렴하고 비교 조건이 일치하는 계산값을 표시합니다. 미완료·미수렴·비교 불가 항목을 0으로 표시하지 않습니다."}
                  {variable === "bond_distance" &&
                    " 점이나 표의 조건을 선택하면 해당 입력 구조를 확인할 수 있습니다. 점 사이의 선은 눈금 안내이며 계산한 중간 구조가 아닙니다."}
                </p>
                <div className="table-wrap lab-table-wrap">
                  <table data-testid="experiment-table">
                    <thead>
                      <tr>
                        <th>{variableLabel}</th>
                        <th>상태</th>
                        <th>기저 함수</th>
                        <th>SCF 반복</th>
                        <th>{crystal ? "셀 에너지 / Eh" : "에너지 / Eh"}</th>
                        <th>ΔE이전 / Eh</th>
                        {crystal && <th>원자당 / Eh</th>}
                        <th>시간 / s</th>
                        <th>peak RSS / MiB</th>
                        <th>확인</th>
                      </tr>
                    </thead>
                    <tbody>
                      {rows.map((row, index) => (
                        <tr
                          key={row.job_id}
                          className={
                            highlight === row.job_id ? "selected-row" : ""
                          }
                        >
                          <th scope="row">
                            <button
                              className="lab-sample-select"
                              aria-label={`${rowLabel(row)} 조건 표시`}
                              aria-pressed={highlight === row.job_id}
                              onClick={() => selectRow(index)}
                            >
                              {rowLabel(row)}
                            </button>
                          </th>
                          <td>
                            <span className={`lab-row-status ${row.status}`}>
                              {statusLabels[row.status] ?? row.status}
                            </span>
                            {row.error && (
                              <details className="lab-row-error">
                                <summary>진단</summary>
                                {row.error}
                              </details>
                            )}
                            {!row.comparable &&
                              TERMINAL.has(row.status) &&
                              row.comparison_note && (
                                <details className="lab-row-error">
                                  <summary>비교 제외 사유</summary>
                                  {row.comparison_note}
                                </details>
                              )}
                          </td>
                          <td>{finite(row.basis_functions) ?? "—"}</td>
                          <td>{finite(row.iteration_count) ?? "—"}</td>
                          <td data-testid="experiment-energy">
                            {row.comparable
                              ? fixed(row.energy_hartree, 9)
                              : "—"}
                          </td>
                          <td>
                            {row.comparable
                              ? fixed(row.delta_previous_hartree, 9)
                              : "—"}
                          </td>
                          {crystal && (
                            <td>
                              {row.comparable
                                ? fixed(row.energy_per_atom_hartree, 9)
                                : "—"}
                            </td>
                          )}
                          <td>{fixed(row.elapsed_seconds)}</td>
                          <td>{fixed(row.peak_memory_mb, 1)}</td>
                          <td>
                            <button
                              className="text-button"
                              aria-label="계산 자세히 보기"
                              title={`${rowLabel(row)}의 실제 계산 결과 확인`}
                              onClick={() => onOpenJob(row.job_id)}
                            >
                              자세히
                              <ArrowRight size={13} />
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
            ) : (
              <div className="lab-empty">
                <strong>계산 결과 대기</strong>
                <p>
                  에너지, 기저 함수 수, 수렴 반복과 계산 비용을 한눈에
                  비교합니다.
                </p>
                <div className="lab-placeholder-bases">
                  {labels.map((label, index) => (
                    <span key={`${label}-${index}`}>{label}</span>
                  ))}
                </div>
              </div>
            )}
          </section>
          <section className="panel lab-interpretation">
            <div className="panel-heading">
              <h2>관찰과 해석</h2>
            </div>
            <div className="lab-findings" aria-live="polite">
              {experiment?.analysis.messages?.length ? (
                experiment.analysis.messages
                  .filter((message) => message.kind !== "limit")
                  .map((message, index) => (
                    <p
                      key={`${index}-${message.kind}`}
                      className={`lab-finding ${message.kind}`}
                    >
                      {message.kind === "next" && <strong>다음 비교 </strong>}
                      {message.text}
                    </p>
                  ))
              ) : (
                <p className="lab-finding">
                  계산 후 실제 에너지 변화와 비교할 수 있는 범위를 표시합니다.
                </p>
              )}
            </div>
            <details className="lab-disclosure lab-limitations">
              <summary>해석 시 확인 사항</summary>
              {experiment?.analysis.messages
                ?.filter((message) => message.kind === "limit")
                .map((message, index) => (
                  <p key={index}>{message.text}</p>
                ))}
              <p>
                SCF 수렴과 기저·격자·k점 수렴은 다릅니다. 서로 다른 근사 방법의
                전체 에너지만으로 정확도 순위를 정하거나 실험값과의 일치를
                보장할 수 없습니다.
              </p>
            </details>
            <details className="lab-disclosure lab-guide">
              <summary>실험 안내와 출처</summary>
              <div className="lab-lessons">
                {(presentation?.lessons ?? []).map((lesson) => (
                  <article key={lesson.title}>
                    <h3>{lesson.title}</h3>
                    <p>{lesson.body}</p>
                  </article>
                ))}
              </div>
              {!!presentation?.references?.length && (
                <div className="lab-references">
                  {presentation.references.map((reference) => (
                    <a
                      key={reference}
                      href={reference}
                      target="_blank"
                      rel="noreferrer"
                    >
                      {reference}
                      <ArrowRight size={12} />
                    </a>
                  ))}
                </div>
              )}
            </details>
          </section>
          {experiment && (
            <footer className="lab-footer">
              <span>
                실험 {experiment.id.slice(0, 8)} · 원본 조건과 결과 보존
              </span>
              <div>
                <a
                  href={`/api/experiments/${experiment.id}/export?format=csv`}
                  download
                >
                  <ArrowDownToLine size={14} />
                  실험 CSV 저장
                </a>
                <a
                  href={`/api/experiments/${experiment.id}/export?format=json`}
                  download
                >
                  <ArrowDownToLine size={14} />
                  실험 JSON 저장
                </a>
              </div>
            </footer>
          )}
        </div>
      </div>
    </main>
  );
}

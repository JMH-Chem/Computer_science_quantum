import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Activity,
  ArrowDownToLine,
  Atom as AtomIcon,
  BookOpen,
  Check,
  ChevronDown,
  CircleHelp,
  Clock3,
  Cpu,
  FlaskConical,
  History,
  Layers3,
  LoaderCircle,
  Maximize2,
  Play,
  Radio,
  Square,
  X,
} from "lucide-react";
import Settings from "./Settings";
import Viewer from "./Viewer";
import {
  Bands,
  Compare,
  Convergence,
  Excitations,
  Learn,
  Orbitals,
} from "./Results";
import Chart from "./Chart";
import { api } from "./api";
import type {
  Catalog,
  Example,
  Job,
  JobEvent,
  OptimizationStep,
  Spec,
  VolumeRequest,
} from "./types";
import { defaultSpec, statusLabels, taskLabels, TERMINAL } from "./types";

type Tab =
  "convergence" | "orbitals" | "excitations" | "bands" | "compare" | "learn";
const emptyCatalog: Catalog = {
  functionals: [],
  bases: [],
  methods: [],
  lessons: [],
};
const tabs: { id: Tab; label: string }[] = [
  { id: "convergence", label: "수렴 과정" },
  { id: "orbitals", label: "전자 구조" },
  { id: "excitations", label: "여기 상태" },
  { id: "bands", label: "밴드 구조" },
  { id: "compare", label: "계산 비교" },
  { id: "learn", label: "함께 배우기" },
];
export default function App() {
  const [spec, setSpec] = useState<Spec>(defaultSpec),
    [examples, setExamples] = useState<Example[]>([]),
    [catalog, setCatalog] = useState<Catalog>(emptyCatalog),
    [jobs, setJobs] = useState<Job[]>([]);
  const [job, setJob] = useState<Job | null>(null),
    [selectedId, setSelectedId] = useState<string | null>(() =>
      localStorage.getItem("quantum-lab.job"),
    );
  const [events, setEvents] = useState<JobEvent[]>([]),
    [tab, setTab] = useState<Tab>("convergence"),
    [expert, setExpert] = useState(false),
    [error, setError] = useState(""),
    [online, setOnline] = useState(false),
    [submitting, setSubmitting] = useState(false),
    [historyOpen, setHistoryOpen] = useState(false);
  const [selectedEvent, setSelectedEvent] = useState<number | null>(null),
    [geometryStep, setGeometryStep] = useState(-1),
    [logsOpen, setLogsOpen] = useState(false);
  const [volumeRequest, setVolumeRequest] = useState<VolumeRequest>({
      kind: "density",
      index: 0,
      spin: 0,
      grid: 40,
    }),
    [volumeJob, setVolumeJob] = useState<Job | null>(null),
    [cube, setCube] = useState<string | null>(null),
    [cubeKind, setCubeKind] = useState("density"),
    [iso, setIso] = useState(0.04),
    [volumeBusy, setVolumeBusy] = useState(false);
  const refreshJobs = useCallback(async () => {
    try {
      setJobs(await api<Job[]>("/jobs"));
    } catch {
      /* selected job fetch reports connection errors */
    }
  }, []);
  useEffect(() => {
    let alive = true;
    Promise.all([
      api<Example[]>("/examples"),
      api<Catalog>("/catalog"),
      api<unknown>("/health"),
    ])
      .then(([e, c]) => {
        if (alive) {
          setExamples(e);
          setCatalog(c);
          setOnline(true);
        }
      })
      .catch((e) => {
        if (alive) setError(`계산 서버에 연결할 수 없습니다. ${String(e)}`);
      });
    void refreshJobs();
    const interval = setInterval(() => {
      void refreshJobs();
      void api("/health")
        .then(() => setOnline(true))
        .catch(() => setOnline(false));
    }, 4000);
    return () => {
      alive = false;
      clearInterval(interval);
    };
  }, [refreshJobs]);
  useEffect(() => {
    setJob(null);
    setEvents([]);
    setSelectedEvent(null);
    setGeometryStep(-1);
    setCube(null);
    setVolumeJob(null);
    setVolumeBusy(false);
    setVolumeRequest({ kind: "density", index: 0, spin: 0, grid: 40 });
    if (!selectedId) {
      setJob(null);
      localStorage.removeItem("quantum-lab.job");
      return;
    }
    localStorage.setItem("quantum-lab.job", selectedId);
    let alive = true;
    let source: EventSource | null = null;
    let initial = true;
    const update = () =>
      api<Job>(`/jobs/${selectedId}`)
        .then((j) => {
          if (alive && initial) {
            initial = false;
            setSpec({ ...j.spec });
          }
          if (alive)
            setJob((previous) =>
              previous?.id === j.id &&
              previous.updated_at === j.updated_at &&
              previous.status === j.status &&
              JSON.stringify(previous.result) === JSON.stringify(j.result)
                ? previous
                : j,
            );
          return j;
        })
        .catch((e) => {
          if (alive) setError(String(e));
        });
    const append = (event: JobEvent) => {
      if (!alive) return;
      setEvents((old) =>
        old.some((e) => e.seq === event.seq)
          ? old
          : [...old, event].sort((a, b) => a.seq - b.seq),
      );
      if (event.type === "status") {
        void update();
        void refreshJobs();
      }
    };
    void api<JobEvent[]>(`/jobs/${selectedId}/history`)
      .then((history) => {
        if (alive)
          setEvents((old) =>
            Array.from(
              new Map([...history, ...old].map((e) => [e.seq, e])).values(),
            ).sort((a, b) => a.seq - b.seq),
          );
      })
      .catch((e) => {
        if (alive) setError(String(e));
      });
    void update().then((j) => {
      if (!alive || !j || TERMINAL.has(j.status)) return;
      source = new EventSource(`/api/jobs/${selectedId}/events`);
      const receive = (e: MessageEvent) => {
        try {
          const event = JSON.parse(e.data) as JobEvent;
          append(event);
          if (
            event.type === "status" &&
            TERMINAL.has(String(event.data.status))
          )
            source?.close();
        } catch {
          /* SSE heartbeat */
        }
      };
      source.onmessage = receive;
      [
        "status",
        "scf_iteration",
        "optimization_step",
        "stage",
        "warning",
      ].forEach((t) => source?.addEventListener(t, receive as EventListener));
    });
    const poll = setInterval(() => {
      void update();
    }, 2500);
    return () => {
      alive = false;
      source?.close();
      clearInterval(poll);
    };
  }, [selectedId, refreshJobs]);
  useEffect(() => {
    if (!volumeJob) return;
    let alive = true;
    let interval: ReturnType<typeof setInterval> | undefined;
    const handle = async (j: Job) => {
      if (!alive) return;
      if (TERMINAL.has(j.status)) {
        if (interval) clearInterval(interval);
        setVolumeBusy(false);
        if (j.status === "succeeded" && j.result?.artifact) {
          const response = await fetch(
            `/api/jobs/${j.id}/artifacts/${encodeURIComponent(j.result.artifact)}`,
          );
          if (!response.ok) throw new Error("격자 파일을 불러오지 못했습니다.");
          const text = await response.text();
          if (alive) {
            setCube(text);
            setCubeKind(j.spec.volume?.kind ?? "density");
            setVolumeJob(j);
          }
        } else setError(j.error ?? "3D 격자 생성이 완료되지 않았습니다.");
      } else setVolumeJob(j);
    };
    const fail = (e: unknown) => {
      if (alive) {
        setVolumeBusy(false);
        setError(String(e));
      }
    };
    if (TERMINAL.has(volumeJob.status)) void handle(volumeJob).catch(fail);
    else
      interval = setInterval(() => {
        void api<Job>(`/jobs/${volumeJob.id}`).then(handle).catch(fail);
      }, 1000);
    return () => {
      alive = false;
      if (interval) clearInterval(interval);
    };
  }, [volumeJob?.id]);
  async function run(s: Spec) {
    setSubmitting(true);
    setError("");
    try {
      const created = await api<Job>("/jobs", {
        method: "POST",
        body: JSON.stringify(s),
      });
      setJob(created);
      setSelectedId(created.id);
      setTab("convergence");
      void refreshJobs();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSubmitting(false);
    }
  }
  async function cancel() {
    if (!job) return;
    try {
      setJob(await api<Job>(`/jobs/${job.id}/cancel`, { method: "POST" }));
      void refreshJobs();
    } catch (e) {
      setError(String(e));
    }
  }
  async function volume() {
    if (!job) return;
    setGeometryStep(-1);
    setVolumeJob(null);
    setVolumeBusy(true);
    setCube(null);
    setError("");
    try {
      const created = await api<Job>(`/jobs/${job.id}/volumes`, {
        method: "POST",
        body: JSON.stringify(volumeRequest),
      });
      setVolumeJob(created);
    } catch (e) {
      setVolumeBusy(false);
      setError(String(e));
    }
  }
  const activeSpec = job?.spec ?? spec,
    result = job?.result;
  const running = !!job && !TERMINAL.has(job.status);
  const iterations = useMemo(
    () => events.filter((e) => e.type === "scf_iteration"),
    [events],
  );
  const optimization = useMemo<OptimizationStep[]>(
    () =>
      result?.optimization ??
      events
        .filter((e) => e.type === "optimization_step")
        .map((e) => e.data as unknown as OptimizationStep),
    [events, result],
  );
  const geometry = geometryStep >= 0 ? optimization[geometryStep] : undefined;
  const atoms = geometry?.atoms ?? result?.atoms ?? activeSpec.atoms,
    lattice = result?.lattice ?? activeSpec.lattice;
  const energy =
    result?.energy_hartree ??
    (iterations.at(-1)?.data.energy_hartree as number | undefined);
  const selectedIteration = useCallback(
    (e: JobEvent) => {
      setSelectedEvent(e.seq);
      if (typeof e.data.optimization_step === "number") {
        const index = optimization.findIndex(
          (s) => s.step === e.data.optimization_step,
        );
        if (index >= 0) {
          setGeometryStep(index);
          setCube(null);
        }
      }
    },
    [optimization],
  );
  const stage = events.filter((e) => e.type === "stage").at(-1)?.data;
  const visibleJobs = jobs.filter((j) => j.spec?.task !== "volume");
  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="#" onClick={(e) => e.preventDefault()}>
          <div className="brand-mark">
            <AtomIcon size={25} />
          </div>
          <div>
            Quantum<span>Lab</span>
            <small>MOLECULES · MATERIALS · DISCOVERY</small>
          </div>
        </a>
        <div className="header-divider" />
        <span className="workspace-name">
          개인 연구실 <ChevronDown size={12} />
        </span>
        <div className="header-right">
          <span className={`connection ${online ? "connected" : ""}`}>
            <i />
            {online ? "로컬 엔진 연결됨" : "연결 확인 중"}
          </span>
          <div className="mode-switch">
            <button
              className={!expert ? "selected" : ""}
              onClick={() => setExpert(false)}
            >
              <BookOpen size={13} />
              학습
            </button>
            <button
              className={expert ? "selected" : ""}
              onClick={() => setExpert(true)}
            >
              <FlaskConical size={13} />
              전문가
            </button>
          </div>
          <button
            className={`icon-button history-toggle ${historyOpen ? "selected" : ""}`}
            aria-label="계산 기록 열기"
            onClick={() => setHistoryOpen((v) => !v)}
          >
            <History size={19} />
          </button>
        </div>
      </header>
      {error && (
        <div className="error-banner" role="alert">
          <CircleHelp size={18} />
          <span>{error}</span>
          <button
            className="icon-button"
            aria-label="알림 닫기"
            onClick={() => setError("")}
          >
            <X size={17} />
          </button>
        </div>
      )}
      <div className="workspace">
        <Settings
          spec={spec}
          setSpec={setSpec}
          examples={examples}
          catalog={catalog}
          expert={expert}
          onRun={(s) => void run(s)}
          busy={submitting}
          onError={setError}
        />
        <main className="main-workspace">
          <div className="workspace-heading">
            <div>
              <div className="breadcrumb">
                WORKSPACE <ChevronDown size={11} />{" "}
                {activeSpec.domain === "crystal"
                  ? "PERIODIC MATERIALS"
                  : "MOLECULAR SCIENCE"}
              </div>
              <h1>{job?.spec.name ?? spec.name}</h1>
              <p>
                {activeSpec.domain === "crystal"
                  ? "주기적 구조에서 전자의 에너지를 탐색합니다."
                  : "전자 구조를 계산하고, 결과에 담긴 의미를 살펴보세요."}
              </p>
            </div>
            <div className="heading-actions">
              {job && (
                <>
                  <button
                    className="secondary-button"
                    onClick={() => {
                      setSpec({ ...job.spec });
                      setSelectedId(null);
                      setJob(null);
                      setTab("convergence");
                    }}
                  >
                    <Play size={13} />
                    조건 복사
                  </button>
                  <a
                    className="secondary-button"
                    href={`/api/jobs/${job.id}/export?format=json`}
                    download
                  >
                    <ArrowDownToLine size={14} />
                    결과 저장
                  </a>
                </>
              )}
            </div>
          </div>
          <div className="metrics-row">
            <div className="metric">
              <span className="metric-label">
                <Activity size={14} /> 전체 에너지
              </span>
              <div className="metric-value" data-testid="energy">
                {typeof energy === "number" ? energy.toFixed(8) : "—"}
                <small>Eh</small>
              </div>
              <span className="metric-detail">
                {result?.energy_hartree !== undefined
                  ? "엔진의 최종 결과"
                  : "SCF 중간값 · 수렴 후 최종값 확정"}
              </span>
            </div>
            <div className="metric">
              <span className="metric-label">
                <AtomIcon size={14} /> 계산 조건
              </span>
              <div className="metric-value compact">
                {activeSpec.method === "DFT"
                  ? activeSpec.functional
                  : activeSpec.task === "educational_rhf"
                    ? "자체 RHF"
                    : "Hartree–Fock"}
              </div>
              <span className="metric-detail">
                {typeof activeSpec.basis === "string"
                  ? activeSpec.basis
                  : "원소별 기저"}{" "}
                · {taskLabels[activeSpec.task]}
              </span>
            </div>
            <div className="metric">
              <span className="metric-label">
                <Radio size={14} /> 실행 상태
              </span>
              <div
                className={`status-value ${job?.status ?? "idle"}`}
                data-testid="job-status"
              >
                {running ? (
                  <LoaderCircle className="spin" size={17} />
                ) : job?.status === "succeeded" ? (
                  <Check size={17} />
                ) : (
                  <span className="status-dot" />
                )}
                {job ? (statusLabels[job.status] ?? job.status) : "계산 준비"}
              </div>
              <span className="metric-detail">
                {running
                  ? String(stage?.message ?? "계산을 수행하고 있습니다")
                  : job?.status === "succeeded"
                    ? "수렴 기준 충족 · 결과 확인 가능"
                    : job
                      ? "중간 결과와 진단을 확인하세요"
                      : "CPU 계산 · 클라우드 전송 없음"}
              </span>
            </div>
          </div>
          {job?.error && (
            <div className="job-error" role="alert">
              <strong>계산 진단</strong>
              {job.error}
            </div>
          )}
          {result?.warnings?.map((w, i) => (
            <div className="warning-note" key={i}>
              {w}
            </div>
          ))}
          <div className="visual-grid">
            <section className="panel structure-panel">
              <div className="panel-heading">
                <div>
                  <span className="eyebrow">
                    {activeSpec.domain === "crystal"
                      ? "CRYSTAL STRUCTURE"
                      : "MOLECULAR STRUCTURE"}
                  </span>
                  <h2>
                    {activeSpec.domain === "crystal"
                      ? "결정 구조"
                      : "분자 구조"}{" "}
                    <span className="count-pill">{atoms.length} ATOMS</span>
                  </h2>
                </div>
                <span className="small-tag">
                  {geometry
                    ? `최적화 ${geometry.step}단계`
                    : result?.atoms
                      ? "최종 구조"
                      : "입력 구조"}
                </span>
              </div>
              <Viewer
                atoms={atoms}
                lattice={lattice}
                cube={cube}
                kind={cubeKind}
                iso={iso}
              />
              {optimization.length > 0 && (
                <div className="geometry-control">
                  <label>
                    최적화 단계 <strong>{geometry?.step ?? "최종"}</strong>
                    <input
                      aria-label="최적화 단계 선택"
                      type="range"
                      min={-1}
                      max={optimization.length - 1}
                      value={geometryStep}
                      onChange={(e) => {
                        setGeometryStep(Number(e.target.value));
                        setCube(null);
                      }}
                    />
                  </label>
                  <small>구조 최적화 단계는 실제 시간과 다릅니다.</small>
                </div>
              )}
              <div className="volume-tools">
                <div className="volume-title">
                  <Maximize2 size={13} />
                  전자밀도와 오비탈<small>수렴 후 격자 생성</small>
                </div>
                <div className="volume-row">
                  <select
                    aria-label="3D 표시 종류"
                    value={volumeRequest.kind}
                    onChange={(e) =>
                      setVolumeRequest({
                        ...volumeRequest,
                        kind: e.target.value as VolumeRequest["kind"],
                        index: 0,
                      })
                    }
                  >
                    <option value="density">전자밀도</option>
                    {activeSpec.domain === "molecule" && (
                      <>
                        <option value="orbital">분자 오비탈</option>
                        {!!result?.excitations?.length && (
                          <>
                            <option value="nto_hole">NTO · hole</option>
                            <option value="nto_particle">NTO · particle</option>
                          </>
                        )}
                      </>
                    )}
                  </select>
                  {volumeRequest.kind !== "density" && (
                    <input
                      type="number"
                      aria-label={
                        volumeRequest.kind === "orbital"
                          ? "오비탈 번호"
                          : "NTO 상태 번호"
                      }
                      title={
                        volumeRequest.kind === "orbital"
                          ? "오비탈 번호 (1부터)"
                          : "여기 상태 번호 (1부터)"
                      }
                      min={1}
                      value={volumeRequest.index + 1}
                      onChange={(e) =>
                        setVolumeRequest({
                          ...volumeRequest,
                          index: Math.max(0, Number(e.target.value) - 1),
                        })
                      }
                    />
                  )}
                  <button
                    className="secondary-button"
                    disabled={job?.status !== "succeeded" || volumeBusy}
                    onClick={() => void volume()}
                  >
                    {volumeBusy ? (
                      <LoaderCircle className="spin" size={13} />
                    ) : (
                      <Layers3 size={13} />
                    )}
                    격자 생성
                  </button>
                </div>
                {expert && (
                  <div className="field-row">
                    <label className="field">
                      격자 해상도
                      <input
                        type="number"
                        min={20}
                        max={80}
                        step={10}
                        value={volumeRequest.grid}
                        onChange={(e) =>
                          setVolumeRequest({
                            ...volumeRequest,
                            grid: Number(e.target.value),
                          })
                        }
                      />
                    </label>
                    {activeSpec.domain === "molecule" &&
                      !activeSpec.restricted && (
                        <label className="field">
                          스핀 채널
                          <select
                            value={volumeRequest.spin}
                            onChange={(e) =>
                              setVolumeRequest({
                                ...volumeRequest,
                                spin: Number(e.target.value),
                              })
                            }
                          >
                            <option value={0}>α</option>
                            <option value={1}>β</option>
                          </select>
                        </label>
                      )}
                  </div>
                )}
                {volumeBusy && (
                  <small className="volume-progress">
                    {volumeJob ? statusLabels[volumeJob.status] : "등록 중"} ·
                    계산 대기열에서 격자를 생성합니다.
                  </small>
                )}
                {cube && (
                  <>
                    <label className="iso-control">
                      등가면 값
                      <input
                        aria-label="등가면 값"
                        type="range"
                        min={0.005}
                        max={0.15}
                        step={0.005}
                        value={iso}
                        onChange={(e) => setIso(Number(e.target.value))}
                      />
                      <span>{iso.toFixed(3)}</span>
                      <button
                        className="text-button"
                        onClick={() => setCube(null)}
                      >
                        숨기기
                      </button>
                    </label>
                    <p className="volume-legend">
                      <i className="positive" />{" "}
                      {cubeKind === "density" ? "양의 밀도" : "양의 위상"}{" "}
                      {cubeKind !== "density" && (
                        <>
                          <i className="negative" /> 음의 위상
                        </>
                      )}
                      <a
                        href={`/api/jobs/${volumeJob?.id}/artifacts/${volumeJob?.result?.artifact}`}
                        download
                      >
                        CUBE 저장
                      </a>
                    </p>
                    {activeSpec.domain === "crystal" && (
                      <small>
                        가상퍼텐셜의 원자가 전자밀도이며 코어 전자를 포함하지
                        않습니다.
                      </small>
                    )}
                  </>
                )}
              </div>
            </section>
            <section className="panel convergence-panel">
              <Convergence
                events={events}
                selected={selectedEvent}
                onSelect={selectedIteration}
              />
              {running && (
                <div className="live-footer">
                  <span>
                    <i className="live-dot" />
                    실제 계산 이벤트를 수신합니다
                  </span>
                  <button className="stop-button" onClick={() => void cancel()}>
                    <Square size={11} fill="currentColor" />
                    계산 취소
                  </button>
                </div>
              )}
              {!running && (
                <div className="live-footer">
                  <span>
                    <Cpu size={13} />
                    {result?.engine
                      ? `${result.engine.name} ${result.engine.version}`
                      : "PySCF · CPU backend"}
                  </span>
                  <button
                    className="text-button"
                    onClick={() => setLogsOpen((v) => !v)}
                  >
                    계산 기록 <ChevronDown size={12} />
                  </button>
                </div>
              )}
              {logsOpen && (
                <div className="event-log">
                  {events.slice(-60).map((e) => (
                    <div key={e.seq}>
                      <span>
                        #{e.seq} {e.type}
                      </span>
                      <code>{JSON.stringify(e.data)}</code>
                    </div>
                  ))}
                  {job && (
                    <a
                      href={`/api/jobs/${job.id}/artifacts/engine.log`}
                      target="_blank"
                      rel="noreferrer"
                    >
                      전체 엔진 로그 열기
                    </a>
                  )}
                </div>
              )}
            </section>
          </div>
          <section className="panel result-panel">
            <nav className="result-tabs" aria-label="결과 탐색">
              {tabs
                .filter((t) =>
                  activeSpec.domain === "crystal"
                    ? !["orbitals", "excitations"].includes(t.id)
                    : t.id !== "bands",
                )
                .map((t) => (
                  <button
                    key={t.id}
                    className={tab === t.id ? "active" : ""}
                    onClick={() => setTab(t.id)}
                  >
                    {t.id === "learn" && <BookOpen size={14} />} {t.label}
                  </button>
                ))}
            </nav>
            <div className="result-content">
              {tab === "convergence" ? (
                <div className="overview">
                  <div>
                    <span className="eyebrow">YOUR NEXT QUESTION</span>
                    <h2>숫자에서 이해로, 한 걸음 더.</h2>
                    <p>
                      계산 조건을 바꾸면 결과는 어떻게 달라질까요?
                      <br />
                      기저와 수치 격자의 수렴을 확인하고, 같은 구조의 결과를
                      비교해 보세요.
                    </p>
                    <button
                      className="text-button"
                      onClick={() => setTab("learn")}
                    >
                      계산 개념 살펴보기 <BookOpen size={14} />
                    </button>
                  </div>
                  <div className="overview-steps">
                    <div>
                      <span>01</span>
                      <strong>구조와 조건</strong>
                      <small>원자 · 전하 · 방법 · 기저</small>
                    </div>
                    <div>
                      <span>02</span>
                      <strong>수렴과 진단</strong>
                      <small>에너지 · 밀도 · 잔차</small>
                    </div>
                    <div>
                      <span>03</span>
                      <strong>결과와 의미</strong>
                      <small>전자 구조 · 스펙트럼 · 밴드</small>
                    </div>
                  </div>
                  {optimization.length > 0 && (
                    <div className="optimization-chart">
                      <Chart
                        data={[
                          {
                            x: optimization.map((s) => s.step),
                            y: optimization.map((s) => s.energy_hartree),
                            type: "scatter",
                            mode: "lines+markers",
                            line: { color: "#b0a1fa" },
                          },
                        ]}
                        xTitle="구조 최적화 단계 (시간 아님)"
                        yTitle="Energy / Eh"
                        onPoint={(i) => {
                          setGeometryStep(i);
                          setCube(null);
                        }}
                      />
                    </div>
                  )}
                </div>
              ) : tab === "orbitals" ? (
                <Orbitals result={result} />
              ) : tab === "excitations" ? (
                <Excitations result={result} />
              ) : tab === "bands" ? (
                <Bands result={result} />
              ) : tab === "compare" ? (
                <Compare jobs={jobs} current={job} />
              ) : (
                <Learn catalog={catalog} result={result} />
              )}
            </div>
          </section>
          {job && (
            <footer className="results-footer">
              <span>
                JOB {job.id.slice(0, 8)} · 모든 위치 단위 Å · 에너지 Eh / eV
              </span>
              <div>
                <a href={`/api/jobs/${job.id}/export?format=csv`} download>
                  반복 CSV
                </a>
                <a
                  href={`/api/jobs/${job.id}/export?format=${activeSpec.domain === "crystal" ? "cif" : "xyz"}`}
                  download
                >
                  {activeSpec.domain === "crystal" ? "CIF" : "XYZ"} 구조
                </a>
                <a
                  href={`/api/jobs/${job.id}/artifacts`}
                  target="_blank"
                  rel="noreferrer"
                >
                  전체 산출물
                </a>
              </div>
            </footer>
          )}
        </main>
      </div>
      {historyOpen && (
        <>
          <div
            className="drawer-backdrop"
            onClick={() => setHistoryOpen(false)}
          />
          <aside className="history-drawer">
            <div className="panel-heading">
              <div>
                <span className="eyebrow">SAVED LOCALLY</span>
                <h2>계산 기록</h2>
              </div>
              <button
                className="icon-button"
                aria-label="계산 기록 닫기"
                onClick={() => setHistoryOpen(false)}
              >
                <X size={18} />
              </button>
            </div>
            <p className="drawer-intro">
              브라우저를 닫아도 계산 기록은 로컬 서버에 보관됩니다.
            </p>
            {visibleJobs.length ? (
              visibleJobs.map((j) => (
                <button
                  className={`history-job ${j.id === selectedId ? "selected" : ""}`}
                  key={j.id}
                  onClick={() => {
                    setSelectedId(j.id);
                    setJob(j);
                    setHistoryOpen(false);
                    setTab("convergence");
                  }}
                >
                  <span className="history-icon">
                    {j.spec.domain === "crystal" ? (
                      <Layers3 size={18} />
                    ) : (
                      <AtomIcon size={18} />
                    )}
                  </span>
                  <span>
                    <strong>{j.spec.name}</strong>
                    <small>
                      {j.created_at
                        ? new Date(j.created_at).toLocaleString("ko-KR")
                        : "로컬 계산"}
                    </small>
                    <em className={j.status}>{statusLabels[j.status]}</em>
                  </span>
                </button>
              ))
            ) : (
              <div className="empty-result">
                <Clock3 size={28} />
                <p>아직 저장된 계산이 없습니다.</p>
              </div>
            )}
          </aside>
        </>
      )}
    </div>
  );
}

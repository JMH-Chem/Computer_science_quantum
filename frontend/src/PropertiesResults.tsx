import { useMemo, useState } from "react";
import { ArrowDownToLine, Pause, Play, Waves } from "lucide-react";
import type { Data } from "plotly.js";
import Chart from "./Chart";
import Viewer from "./Viewer";
import { Empty } from "./Results";
import type { Atom, CorrelationIteration, JobEvent, Result } from "./types";
import "./properties.css";

const EH_TO_KJ_MOL = 2625.4996394799;
const number = (value: unknown, digits = 5) =>
  typeof value === "number" && Number.isFinite(value)
    ? value.toFixed(digits)
    : "—";

export function DataDownload({
  jobId,
  kind,
  label = "데이터 CSV",
}: {
  jobId?: string;
  kind: string;
  label?: string;
}) {
  return jobId ? (
    <a
      className="text-button"
      href={`/api/jobs/${jobId}/data?kind=${kind}`}
      download
    >
      <ArrowDownToLine size={13} />
      {label}
    </a>
  ) : null;
}

export function CorrelationResults({
  result,
  events,
  jobId,
}: {
  result?: Result | null;
  events: JobEvent[];
  jobId?: string;
}) {
  const correlation = result?.correlation;
  const iterations = useMemo(
    () =>
      correlation?.iterations ??
      events
        .filter((event) => event.type === "correlation_iteration")
        .map((event) => event.data as unknown as CorrelationIteration),
    [correlation, events],
  );
  const data = useMemo<Data[]>(
    () => [
      {
        x: iterations.map((entry) => entry.iteration),
        y: iterations.map((entry) => entry.energy_hartree),
        type: "scatter",
        mode: "lines+markers",
        line: { color: "#b7a4f4" },
        connectgaps: false,
      },
    ],
    [iterations],
  );
  if (!correlation && !iterations.length)
    return (
      <Empty
        label="전자 상관 계산"
        description="MP2, CCSD, CCSD(T), CISD, FCI, CASCI 또는 CASSCF 단일점 계산을 실행하면 기준 RHF와 상관 에너지, 수렴 정보를 표시합니다."
      />
    );
  return (
    <div className="advanced-results" data-testid="correlation-results">
      <div className="section-header">
        <div>
          <h2>
            {correlation?.method ?? "전자 상관"} · 기준 상태에서 상관 에너지로
          </h2>
          <p>
            SCF와 상관 풀이기의 반복은 별도 과정입니다. 반복 번호는 실제 시간이
            아닙니다.
          </p>
        </div>
        {correlation && <DataDownload jobId={jobId} kind="correlation" />}
      </div>
      {correlation && (
        <>
          <div className="property-cards">
            <div>
              <span>RHF 기준 에너지 / Eh</span>
              <strong>{number(correlation.reference_energy_hartree, 9)}</strong>
            </div>
            <div>
              <span>상관 에너지 / Eh</span>
              <strong>
                {number(correlation.correlation_energy_hartree, 9)}
              </strong>
            </div>
            <div>
              <span>
                {correlation.converged
                  ? "최종 전체 에너지 / Eh"
                  : "미수렴 중간 에너지 / Eh"}
              </span>
              <strong>{number(correlation.total_energy_hartree, 9)}</strong>
            </div>
            <div>
              <span>상관 풀이기 수렴</span>
              <strong>{correlation.converged ? "수렴" : "미수렴"}</strong>
            </div>
          </div>
          <p className="science-note">
            이 작업의 전자밀도·오비탈·쌍극자는 RHF 기준 상태입니다. 상관
            에너지로부터 상관 밀도를 계산했다고 해석하지 않습니다. 에너지가
            낮아졌다는 사실만으로 해당 방법의 정확도를 확정할 수 없습니다.
          </p>
          {correlation.frozen_orbitals > 0 && (
            <p>동결 점유 공간 오비탈: {correlation.frozen_orbitals}개</p>
          )}
          {correlation.active_space && (
            <div className="science-note">
              <strong>
                활성 공간 CAS({correlation.active_space.electrons},{" "}
                {correlation.active_space.orbitals})
              </strong>
              <p>
                닫힌껍질 코어 오비탈 {correlation.active_space.core_orbitals}개.
                초기 활성 RHF 오비탈 번호:{" "}
                {correlation.active_space.initial_orbital_indices_1based?.join(
                  ", ",
                ) ?? "계산 기록 확인"}
                . CASSCF에서는 오비탈도 최적화합니다. 선택한 공간의 화학적
                적절성은 별도로 확인해야 합니다.
              </p>
            </div>
          )}
        </>
      )}
      {iterations.length > 0 && (
        <Chart
          data={data}
          xTitle="상관 풀이기 반복 (시간 아님)"
          yTitle="전체 에너지 / Eh"
        />
      )}
      {!!correlation?.natural_occupations?.length && (
        <div>
          <h3>
            자연 오비탈 점유수 ·{" "}
            {correlation.natural_occupations_space === "active"
              ? "활성 공간"
              : "계산 공간"}
          </h3>
          <p>
            정수 점유에서 벗어난 정도를 관찰할 수 있습니다. 이 값만으로
            단일참조·다중참조 방법의 적합성을 판정하지 않습니다.
          </p>
          <div className="occupation-list">
            {correlation.natural_occupations.map((occupation, index) => (
              <span key={index}>
                NO {index + 1}
                <strong>{number(occupation, 5)}</strong>
              </span>
            ))}
          </div>
        </div>
      )}
      {correlation && Object.keys(correlation.diagnostics).length > 0 && (
        <details className="scientific-details">
          <summary>풀이기 진단과 자원 설정</summary>
          <dl>
            {Object.entries(correlation.diagnostics).map(([key, value]) => (
              <div key={key}>
                <dt>{key}</dt>
                <dd>
                  {typeof value === "number"
                    ? number(value, 8)
                    : typeof value === "string" || typeof value === "boolean"
                      ? String(value)
                      : JSON.stringify(value)}
                </dd>
              </div>
            ))}
          </dl>
        </details>
      )}
    </div>
  );
}

export function MolecularPropertyResults({
  result,
  atoms,
  jobId,
}: {
  result?: Result | null;
  atoms: Atom[];
  jobId?: string;
}) {
  const properties = result?.properties;
  const data = useMemo<Data[]>(
    () =>
      properties
        ? [
            {
              x: atoms.map((atom, index) => `${atom.symbol}${index + 1}`),
              y: properties.mulliken_charges,
              name: "Mulliken",
              type: "bar",
              marker: { color: "#68dfc5" },
            },
            {
              x: atoms.map((atom, index) => `${atom.symbol}${index + 1}`),
              y: properties.lowdin_charges,
              name: "Löwdin",
              type: "bar",
              marker: { color: "#b7a4f4" },
            },
          ]
        : [],
    [properties, atoms],
  );
  if (!properties)
    return (
      <Empty
        label="분자의 전하와 쌍극자"
        description="새 분자 계산을 완료하면 Mulliken·Löwdin 전하, 쌍극자와 스핀 진단을 확인할 수 있습니다. 원자 전하는 밀도를 나누는 방식과 기저에 의존합니다."
      />
    );
  return (
    <div className="advanced-results" data-testid="properties-results">
      <div className="section-header">
        <div>
          <h2>분자 물성 · {properties.reference_model}</h2>
          <p>
            동일한 밀도에서도 분할 방식에 따라 원자 전하가 다르게 나타날 수
            있습니다.
          </p>
        </div>
        <DataDownload jobId={jobId} kind="charges" />
      </div>
      <div className="property-cards">
        <div>
          <span>쌍극자 크기 / Debye</span>
          <strong>{number(properties.dipole_magnitude_debye, 6)}</strong>
        </div>
        <div>
          <span>쌍극자 x, y, z / Debye</span>
          <strong className="vector-value">
            {properties.dipole_debye
              .map((value) => number(value, 4))
              .join(" · ")}
          </strong>
        </div>
        <div>
          <span>⟨S²⟩</span>
          <strong>{number(properties.spin_square, 6)}</strong>
        </div>
      </div>
      <p className="science-note">
        Mulliken·Löwdin 전하는 관측 가능한 고유한 원자 전하가 아닙니다. 음수는
        전자 과잉, 양수는 전자 부족을 뜻합니다. 오비탈 등가면의 위상 색상과는
        다른 의미입니다. 이온의 쌍극자는 좌표 원점에 의존합니다.
      </p>
      <Chart
        data={data}
        xTitle="원자"
        yTitle="부분 전하 / e"
        layout={{
          showlegend: true,
          barmode: "group",
          legend: { orientation: "h", y: 1.15 },
        }}
      />
      <div className="table-wrap">
        <table data-testid="atomic-charges">
          <thead>
            <tr>
              <th>원자</th>
              <th>Mulliken / e</th>
              <th>Löwdin / e</th>
            </tr>
          </thead>
          <tbody>
            {atoms.map((atom, index) => (
              <tr key={index}>
                <th>
                  {atom.symbol} {index + 1}
                </th>
                <td>{number(properties.mulliken_charges[index], 7)}</td>
                <td>{number(properties.lowdin_charges[index], 7)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {properties.warnings.map((warning, index) => (
        <p className="science-note" key={index}>
          {warning}
        </p>
      ))}
    </div>
  );
}

export function VibrationalResults({
  result,
  atoms,
  jobId,
}: {
  result?: Result | null;
  atoms: Atom[];
  jobId?: string;
}) {
  const vibrations = result?.vibrations;
  const [modeIndex, setModeIndex] = useState(0);
  const [animate, setAnimate] = useState(false);
  const [amplitude, setAmplitude] = useState(0.3);
  const [width, setWidth] = useState(20);
  const modes = vibrations?.modes ?? [];
  const mode = modes[Math.min(modeIndex, Math.max(modes.length - 1, 0))];
  const spectrum = useMemo<Data[]>(() => {
    if (!vibrations?.ir.computed) return [];
    const positive = vibrations.modes.filter(
      (item) => item.frequency_cm1 > 0 && item.ir_intensity_km_mol !== null,
    );
    const max =
      Math.max(...positive.map((item) => item.frequency_cm1), 500) + width * 5;
    const points = Math.max(700, Math.ceil(max / (width / 8)) + 1);
    const x = Array.from(
      { length: points },
      (_, index) => (index * max) / (points - 1),
    );
    return [
      {
        x,
        y: x.map((frequency) =>
          positive.reduce(
            (sum, item) =>
              sum +
              ((item.ir_intensity_km_mol ?? 0) *
                Math.exp(
                  -0.5 * ((frequency - item.frequency_cm1) / width) ** 2,
                )) /
                (width * Math.sqrt(2 * Math.PI)),
            0,
          ),
        ),
        type: "scatter",
        mode: "lines",
        fill: "tozeroy",
        line: { color: "#68dfc5" },
      },
    ];
  }, [vibrations, width]);
  if (!vibrations)
    return (
      <Empty
        label="진동수와 분자 운동의 형태"
        description="진동수 작업을 실행하면 실제 Hessian의 조화 진동 모드와 선택한 IR 세기를 확인할 수 있습니다. 구조 최적화 결과에서 시작하면 열화학 해석에도 도움이 됩니다."
      />
    );
  const thermo = vibrations.thermochemistry;
  return (
    <div className="advanced-results" data-testid="vibrational-results">
      <div className="section-header">
        <div>
          <h2>조화 진동 · Harmonic vibrations</h2>
          <p>
            병진·회전을 제거한 진동 모드입니다. 음수로 표시한 진동수는 허수
            진동수입니다.
          </p>
        </div>
        <DataDownload jobId={jobId} kind="frequencies" />
      </div>
      <div className="property-cards">
        <div>
          <span>진동 모드</span>
          <strong>{modes.length}개</strong>
        </div>
        <div>
          <span>허수 모드</span>
          <strong>{vibrations.imaginary_modes}개</strong>
        </div>
        <div>
          <span>최대 기울기 / Eh bohr⁻¹</span>
          <strong>{number(vibrations.gradient_max_hartree_per_bohr, 7)}</strong>
        </div>
        <div>
          <span>영점 에너지 / Eh</span>
          <strong>{number(vibrations.zero_point_energy_hartree, 8)}</strong>
        </div>
      </div>
      {!vibrations.stationary && (
        <p className="science-note warning">
          현재 구조는 기울기 기준을 만족하는 정상점이 아닙니다. 최적화 후 다시
          분석하세요. 이 결과를 안정한 분자의 진동수·열화학 값으로 확정하지
          않습니다.
        </p>
      )}
      {modes.length > 0 && (
        <div className="vibration-layout">
          <div className="vibration-view">
            <Viewer
              atoms={atoms}
              modeDisplacements={mode?.displacements}
              animateMode={animate}
              modeAmplitude={amplitude}
            />
            <div className="mode-controls">
              <button
                className="secondary-button"
                onClick={() => setAnimate((value) => !value)}
              >
                {animate ? <Pause size={14} /> : <Play size={14} />}
                {animate ? "모드 정지" : "모드 모식도 재생"}
              </button>
              <label>
                표시 진폭 / Å
                <input
                  aria-label="진동 표시 진폭"
                  type="range"
                  min={0.05}
                  max={0.8}
                  step={0.05}
                  value={amplitude}
                  onChange={(event) => setAmplitude(Number(event.target.value))}
                />
                <span>{amplitude.toFixed(2)}</span>
              </label>
            </div>
            <p className="science-note">
              <Waves size={15} />
              계산된 모드 방향을 사용한 교육용 모식도입니다. 재생 속도와 진폭은
              보기 위한 설정이며 실제 시간 궤적·열적 진폭을 계산한 것이
              아닙니다.
              {mode?.imaginary &&
                " 허수 모드의 왕복 재생은 안정한 진동을 의미하지 않습니다."}
            </p>
          </div>
          <div className="table-wrap mode-table">
            <table data-testid="vibration-modes">
              <thead>
                <tr>
                  <th>모드</th>
                  <th>진동수 / cm⁻¹</th>
                  <th>환산 질량 / amu</th>
                  <th>IR / km mol⁻¹</th>
                </tr>
              </thead>
              <tbody>
                {modes.map((item, index) => (
                  <tr
                    key={item.index}
                    className={index === modeIndex ? "selected-row" : ""}
                  >
                    <th>
                      <button
                        className="text-button"
                        aria-label={`진동 모드 ${index + 1} 선택`}
                        aria-pressed={index === modeIndex}
                        onClick={() => setModeIndex(index)}
                      >
                        {index + 1}
                      </button>
                    </th>
                    <td>
                      {number(item.frequency_cm1, 2)}
                      {item.imaginary ? " (허수)" : ""}
                    </td>
                    <td>{number(item.reduced_mass_amu, 4)}</td>
                    <td>
                      {vibrations.ir.computed
                        ? number(item.ir_intensity_km_mol, 5)
                        : "미계산"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
      {vibrations.ir.computed ? (
        <div>
          <div className="section-header">
            <div>
              <h3>계산한 IR 스펙트럼</h3>
              <p>
                쌍극자의 유한차분 미분으로 세기를 계산했습니다. Gaussian 폭은
                표시 설정이며 고유 진동수와 적분 세기는 그대로입니다.
              </p>
            </div>
            <label className="field">
              표시 Gaussian σ / cm⁻¹
              <input
                aria-label="IR 표시 폭"
                type="number"
                min={5}
                max={200}
                value={width}
                onChange={(event) =>
                  setWidth(
                    Math.max(
                      5,
                      Math.min(200, Number(event.target.value) || 20),
                    ),
                  )
                }
              />
            </label>
          </div>
          <Chart
            data={spectrum}
            xTitle="진동수 / cm⁻¹"
            yTitle="IR 세기 밀도 / (km mol⁻¹)/(cm⁻¹)"
          />
          <p>
            유한차분 변위 {vibrations.ir.finite_difference_step_bohr} bohr ·
            변위 구조 SCF {vibrations.ir.displaced_calculations}회
          </p>
        </div>
      ) : (
        <p className="science-note">
          IR 세기는{" "}
          {vibrations.ir.requested
            ? "계산 요청을 완료하지 못했습니다. 진단을 확인하세요."
            : "요청하지 않아 계산하지 않았습니다. IR 옵션을 켜 새 작업을 실행할 수 있습니다."}
        </p>
      )}
      <div>
        <h3>열화학 · 이상기체 RRHO</h3>
        {thermo ? (
          <>
            <p>
              {thermo.temperature_kelvin} K · {thermo.pressure_pa} Pa · 진동수
              보정 계수 {thermo.frequency_scaling_factor} · 회전 대칭수{" "}
              {thermo.rotational_symmetry_number}
            </p>
            <div className="table-wrap">
              <table data-testid="thermochemistry">
                <thead>
                  <tr>
                    <th>물성</th>
                    <th>값</th>
                    <th>단위</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <th>내부 에너지 U</th>
                    <td>{number(thermo.internal_energy_hartree, 9)}</td>
                    <td>Eh</td>
                  </tr>
                  <tr>
                    <th>엔탈피 H</th>
                    <td>{number(thermo.enthalpy_hartree, 9)}</td>
                    <td>Eh</td>
                  </tr>
                  <tr>
                    <th>Gibbs 에너지 G</th>
                    <td>{number(thermo.gibbs_energy_hartree, 9)}</td>
                    <td>Eh</td>
                  </tr>
                  <tr>
                    <th>엔트로피 S</th>
                    <td>
                      {number(
                        thermo.entropy_hartree_per_kelvin * EH_TO_KJ_MOL * 1000,
                        4,
                      )}
                    </td>
                    <td>J mol⁻¹ K⁻¹</td>
                  </tr>
                  <tr>
                    <th>정적 열용량 Cᵥ</th>
                    <td>
                      {number(
                        thermo.heat_capacity_cv_hartree_per_kelvin *
                          EH_TO_KJ_MOL *
                          1000,
                        4,
                      )}
                    </td>
                    <td>J mol⁻¹ K⁻¹</td>
                  </tr>
                  <tr>
                    <th>정압 열용량 Cₚ</th>
                    <td>
                      {number(
                        thermo.heat_capacity_cp_hartree_per_kelvin *
                          EH_TO_KJ_MOL *
                          1000,
                        4,
                      )}
                    </td>
                    <td>J mol⁻¹ K⁻¹</td>
                  </tr>
                </tbody>
              </table>
            </div>
            <p className="science-note">
              전자 에너지에 이상기체·강체 회전자·조화 진동 기여를 더한 값입니다.
              용매, 비조화성, 저주파 내부 회전, 표준 상태 변환을 자동 포함하지
              않습니다.
            </p>
          </>
        ) : (
          <p className="science-note">
            현재 구조·모드에서 유효한 열화학 값을 제공할 수 없습니다. 정상점과
            양의 진동 모드가 확인되어야 합니다.
          </p>
        )}
      </div>
      {vibrations.warnings.map((warning, index) => (
        <p className="science-note" key={index}>
          {warning}
        </p>
      ))}
    </div>
  );
}

export function MaterialResults({
  result,
  jobId,
}: {
  result?: Result | null;
  jobId?: string;
}) {
  const dos = result?.sampled_dos;
  const properties = result?.crystal_properties;
  const edges = result?.sampled_band_edges;
  const data = useMemo<Data[]>(
    () =>
      dos
        ? [
            {
              x: dos.energy_ev,
              y: dos.density_states_per_ev,
              name: "전체 상태",
              type: "scatter",
              mode: "lines",
              line: { color: "#b7a4f4" },
            },
            {
              x: dos.energy_ev,
              y: dos.occupied_density_states_per_ev,
              name: "점유 상태",
              type: "scatter",
              mode: "lines",
              fill: "tozeroy",
              line: { color: "#68dfc5" },
            },
          ]
        : [],
    [dos],
  );
  if (!dos && !properties)
    return (
      <Empty
        label="상태 밀도와 결정 물성"
        description="새 결정 계산을 완료하면 SCF k점 격자의 Gaussian 상태 밀도, 표본 밴드 끝점과 단위격자 물성을 확인할 수 있습니다."
      />
    );
  return (
    <div className="advanced-results" data-testid="material-results">
      <div className="section-header">
        <div>
          <h2>상태 밀도 · Density of states</h2>
          <p>
            밴드 경로가 아닌 균일 SCF k점 격자의 고유값과 점유수를 사용합니다.
          </p>
        </div>
        {dos && <DataDownload jobId={jobId} kind="dos" />}
      </div>
      {properties && (
        <div className="property-cards">
          <div>
            <span>원자당 에너지 / Eh</span>
            <strong>{number(properties.energy_per_atom_hartree, 8)}</strong>
          </div>
          <div>
            <span>단위격자 부피 / Å³</span>
            <strong>{number(properties.volume_angstrom3, 4)}</strong>
          </div>
          <div>
            <span>질량 밀도 / g cm⁻³</span>
            <strong>{number(properties.density_g_cm3, 4)}</strong>
          </div>
          <div>
            <span>셀 내 원자</span>
            <strong>{properties.atom_count}개</strong>
          </div>
        </div>
      )}
      {dos && (
        <>
          <Chart
            data={data}
            xTitle="에너지 − SCF 격자 VBM / eV"
            yTitle="DOS / states (eV cell)⁻¹"
            layout={{ showlegend: true, legend: { orientation: "h", y: 1.15 } }}
          />
          <p>
            k점 {dos.kpoint_count}개 · Gaussian σ = {dos.width_ev} eV · 스핀
            축퇴도 {dos.spin_degeneracy}
          </p>
          <div className="property-cards">
            <div>
              <span>상태 밀도 적분 / cell</span>
              <strong>
                {number(dos.state_integral, 5)} / {dos.expected_states}
              </strong>
            </div>
            <div>
              <span>점유 상태 적분 / electrons cell⁻¹</span>
              <strong>
                {number(dos.electron_integral, 5)} / {dos.expected_electrons}
              </strong>
            </div>
          </div>
          {dos.limitations.map((note, index) => (
            <p className="science-note" key={index}>
              {note}
            </p>
          ))}
        </>
      )}
      {edges && (
        <div>
          <h3>계산한 점에서의 밴드 끝점</h3>
          <div className="property-cards">
            <div>
              <span>표본 기본 갭 / eV</span>
              <strong>{number(edges.indirect_gap_ev, 6)}</strong>
            </div>
            <div>
              <span>표본 최소 직접 갭 / eV</span>
              <strong>{number(edges.direct_gap_ev, 6)}</strong>
            </div>
          </div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>끝점</th>
                  <th>분수 k 좌표</th>
                  <th>표본</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <th>VBM</th>
                  <td>
                    {edges.vbm.kpoint_scaled
                      .map((n) => number(n, 4))
                      .join(", ")}
                  </td>
                  <td>
                    {edges.vbm.source === "band_path"
                      ? "밴드 경로"
                      : "SCF 격자"}
                  </td>
                </tr>
                <tr>
                  <th>CBM</th>
                  <td>
                    {edges.cbm.kpoint_scaled
                      .map((n) => number(n, 4))
                      .join(", ")}
                  </td>
                  <td>
                    {edges.cbm.source === "band_path"
                      ? "밴드 경로"
                      : "SCF 격자"}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
          <p className="science-note">
            {edges.classification === "direct on sampled points"
              ? "표본에서 직접 갭으로 분류했습니다."
              : "표본에서 간접 갭으로 분류했습니다."}{" "}
            전체 Brillouin zone의 갭이나 광학 여기 에너지로 보장하지 않습니다.
            DOS는 SCF 격자 VBM을 0으로 사용하므로 밴드 경로 그림의 에너지 기준과
            다를 수 있습니다.
          </p>
        </div>
      )}
    </div>
  );
}

export function EOMResults({
  result,
  jobId,
}: {
  result?: Result | null;
  jobId?: string;
}) {
  const eom = result?.eom;
  const data = useMemo<Data[]>(
    () =>
      eom
        ? [
            {
              x: [-0.3, 0.3],
              y: [0, 0],
              type: "scatter",
              mode: "lines",
              line: { color: "#7f98a7", width: 3 },
              name: "CCSD 기준",
            },
            ...eom.states
              .filter((state) => state.converged)
              .map((state): Data => ({
                x: [0.7, 1.3],
                y: [state.energy_ev, state.energy_ev],
                type: "scatter",
                mode: "lines",
                line: { color: "#b7a4f4", width: 3 },
                text: [`S${state.state}`, `S${state.state}`],
                hovertemplate: `S${state.state}: %{y:.6f} eV<extra></extra>`,
              })),
          ]
        : [],
    [eom],
  );
  if (!eom)
    return (
      <Empty
        label="EOM-CCSD 여기 상태"
        description="CCSD 기준에서 수직 singlet 여기 에너지를 계산합니다."
      />
    );
  return (
    <div className="advanced-results" data-testid="eom-results">
      <div className="section-header">
        <div>
          <h2>EOM-CCSD · 수직 singlet 여기 에너지</h2>
          <p>
            기준 상태: {eom.reference} · 요청 상태 {eom.requested_states}개 ·
            잔차 기준 {eom.residual_tolerance.toExponential(1)}
          </p>
        </div>
        <DataDownload jobId={jobId} kind="eom" />
      </div>
      <Chart
        data={data}
        xTitle="CCSD 기준 상태 / EOM 여기 상태"
        yTitle="수직 여기 에너지 / eV"
      />
      <div className="table-wrap">
        <table data-testid="eom-states">
          <thead>
            <tr>
              <th>상태</th>
              <th>여기 에너지 / eV</th>
              <th>잔차 노름</th>
              <th>수렴</th>
            </tr>
          </thead>
          <tbody>
            {eom.states.map((state) => (
              <tr key={state.state}>
                <th>S{state.state}</th>
                <td>{number(state.energy_ev, 7)}</td>
                <td>{state.residual_norm?.toExponential(3) ?? "—"}</td>
                <td>{state.converged ? "수렴" : "미수렴"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="science-note">
        여기 에너지만 계산합니다. oscillator strength·흡수 스펙트럼·NTO와 상태
        수명은 계산하지 않았습니다. 위 전체 에너지는 CCSD 바닥상태 에너지입니다.
      </p>
      {eom.notes.map((note, index) => (
        <p className="science-note" key={index}>
          {note}
        </p>
      ))}
    </div>
  );
}

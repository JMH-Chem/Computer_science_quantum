import { expect, test, type Page } from "@playwright/test";
import { readFile } from "node:fs/promises";

async function openWorkbench(page: Page) {
  await page.goto("/");
  await expect(page.getByText("로컬 엔진 연결됨")).toBeVisible();
  await page.getByRole("button", { name: "계산 작업", exact: true }).click();
}
async function finishedJob(page: Page) {
  await expect(page.getByTestId("job-status")).toContainText("수렴 완료", {
    timeout: 150000,
  });
  const id = await page.evaluate(() => localStorage.getItem("quantum-lab.job"));
  expect(id).toBeTruthy();
  const response = await page.request.get(`/api/jobs/${id}`);
  expect(response.ok()).toBeTruthy();
  return response.json();
}
async function downloadText(page: Page, label: string) {
  const promise = page.waitForEvent("download");
  await page.getByRole("link", { name: label, exact: true }).click();
  const download = await promise;
  return readFile((await download.path())!, "utf8");
}

test("advanced molecule: actual MP2, reference properties, report and reproducible archive", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await openWorkbench(page);
  await page.getByLabel("예제 선택").selectOption("water_mp2");
  await page.getByLabel("계산 이름").fill("물 MP2 <보고서 검증>");
  await expect(page.getByLabel("방법론")).toHaveValue("MP2");
  await page.getByRole("button", { name: "계산 시작" }).click();
  const job = await finishedJob(page);
  expect(job.result.correlation.converged).toBe(true);
  expect(job.result.correlation_energy_hartree).toBeLessThan(0);
  expect(job.result.energy_hartree).toBeCloseTo(
    job.result.reference_energy_hartree + job.result.correlation_energy_hartree,
    10,
  );
  await page.getByRole("button", { name: "전자 상관", exact: true }).click();
  await expect(page.getByTestId("correlation-results")).toContainText(
    job.result.correlation_energy_hartree.toFixed(9),
  );
  await expect(page.locator(".volume-tools")).toContainText("RHF");
  await page.getByRole("button", { name: "물성·전하", exact: true }).click();
  await expect(
    page.getByTestId("atomic-charges").locator("tbody tr"),
  ).toHaveCount(3);
  expect(
    job.result.properties.mulliken_charges.reduce(
      (a: number, b: number) => a + b,
      0,
    ),
  ).toBeCloseTo(0, 8);
  const charges = await downloadText(page, "데이터 CSV");
  expect(charges).toContain("mulliken");
  const popup = page.waitForEvent("popup");
  await page.getByRole("link", { name: "계산 보고서", exact: true }).click();
  const report = await popup;
  await expect(
    report.getByText("물 MP2 <보고서 검증>", { exact: false }).first(),
  ).toBeVisible();
  expect(await report.locator("script").count()).toBe(0);
  await report.close();
  const archivePromise = page.waitForEvent("download");
  await page.getByRole("link", { name: "재현 자료 ZIP", exact: true }).click();
  const archive = await archivePromise;
  expect(archive.suggestedFilename()).toContain(".zip");
  const bytes = await readFile((await archive.path())!);
  expect(bytes.subarray(0, 2).toString()).toBe("PK");
  await page.reload();
  await page.getByRole("button", { name: "전자 상관", exact: true }).click();
  await expect(page.getByTestId("correlation-results")).toBeVisible();
  await page.screenshot({
    path: "test-results/correlation-workspace.png",
    fullPage: true,
  });
  expect(errors).toEqual([]);
});

test("vibrations: optimized water becomes a new input, actual Hessian IR and mode illustration", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await openWorkbench(page);
  await page.getByLabel("예제 선택").selectOption("water_vibrations");
  await page.getByLabel("계산 작업").selectOption("optimize");
  await page.getByRole("button", { name: "계산 시작" }).click();
  const optimized = await finishedJob(page);
  expect(optimized.result.geometry_converged).toBe(true);
  await page
    .getByRole("button", { name: "최종 구조로 새 계산", exact: true })
    .click();
  await expect(page.getByTestId("job-status")).toContainText("계산 준비");
  await page.getByLabel("계산 작업").selectOption("frequencies");
  await page.getByLabel("IR 세기 계산", { exact: false }).check();
  await page.getByRole("button", { name: "계산 시작" }).click();
  const job = await finishedJob(page);
  expect(job.id).not.toBe(optimized.id);
  expect(job.spec.atoms).toEqual(optimized.result.atoms);
  expect(job.result.vibrations.modes).toHaveLength(3);
  expect(job.result.vibrations.ir.computed).toBe(true);
  expect(job.result.vibrations.stationary).toBe(true);
  expect(job.result.vibrations.thermochemistry).toBeTruthy();
  await expect(page.locator(".volume-tools")).not.toContainText(
    "전체 상관 에너지와 계산 수준이 다릅니다",
  );
  await page
    .getByRole("button", { name: "진동·IR·열화학", exact: true })
    .click();
  await expect(
    page.getByTestId("vibration-modes").locator("tbody tr"),
  ).toHaveCount(3);
  await expect(page.getByTestId("thermochemistry")).toBeVisible();
  await expect(page.getByTestId("vibrational-results")).toContainText(
    job.result.vibrations.modes[0].frequency_cm1.toFixed(2),
  );
  await page
    .getByRole("button", { name: "진동 모드 2 선택", exact: true })
    .click();
  await expect(
    page.getByRole("button", { name: "진동 모드 2 선택", exact: true }),
  ).toHaveAttribute("aria-pressed", "true");
  await page
    .getByRole("button", { name: "모드 모식도 재생", exact: true })
    .click();
  await expect(
    page.getByRole("button", { name: "모드 정지", exact: true }),
  ).toBeVisible();
  await expect(page.locator(".vibration-view canvas")).toBeVisible();
  await expect(page.locator(".viewer-error")).toHaveCount(0);
  await page.getByLabel("IR 표시 폭").fill("35");
  const csv = await downloadText(page, "데이터 CSV");
  expect(csv).toContain("frequency");
  await page.screenshot({
    path: "test-results/vibrations-workspace.png",
    fullPage: true,
  });
  await page.getByRole("button", { name: "모드 정지", exact: true }).click();
  expect(errors).toEqual([]);
});

test("materials: actual diamond DOS has correct states, band-edge labels and CSV", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await openWorkbench(page);
  await page.getByRole("button", { name: "재료", exact: true }).click();
  await page.getByLabel("예제 선택").selectOption("diamond");
  await page.getByRole("button", { name: "계산 시작" }).click();
  const job = await finishedJob(page);
  const dos = job.result.sampled_dos;
  expect(dos.state_integral).toBeCloseTo(dos.expected_states, 4);
  expect(dos.electron_integral).toBeCloseTo(dos.expected_electrons, 4);
  await page
    .getByRole("button", { name: "DOS·결정 물성", exact: true })
    .click();
  await expect(page.getByTestId("material-results")).toContainText(
    job.result.crystal_properties.volume_angstrom3.toFixed(4),
  );
  const trace = await page
    .locator(".advanced-results .js-plotly-plot")
    .evaluate((element) => {
      const data = (
        element as HTMLElement & { data: { x: number[]; y: number[] }[] }
      ).data;
      return { x: data[0].x, y: data[0].y };
    });
  expect(trace.x).toEqual(dos.energy_ev);
  expect(trace.y).toEqual(dos.density_states_per_ev);
  const csv = await downloadText(page, "데이터 CSV");
  expect(csv).toContain("energy_ev");
  await page.screenshot({
    path: "test-results/dos-workspace.png",
    fullPage: true,
  });
  expect(errors).toEqual([]);
});

test("active space and EOM: real NEVPT2 and EOM-CCSD with no invented absorption", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await openWorkbench(page);
  await page.getByLabel("예제 선택").selectOption("h2");
  await page.getByLabel("방법론").selectOption("NEVPT2");
  await expect(page.getByLabel("활성 전자 수")).toHaveValue("2");
  await page.getByRole("button", { name: "계산 시작" }).click();
  const nevpt = await finishedJob(page);
  expect(nevpt.result.correlation.method).toBe("NEVPT2");
  await page.getByRole("button", { name: "전자 상관", exact: true }).click();
  await expect(page.getByTestId("correlation-results")).toContainText(
    "CAS(2, 2)",
  );
  await page.getByRole("button", { name: "조건 복사", exact: true }).click();
  await page.getByLabel("계산 작업").selectOption("eomccsd");
  await expect(page.getByLabel("방법론")).toHaveValue("CCSD");
  await page.getByLabel("계산할 여기 상태 수").fill("1");
  await page.getByRole("button", { name: "계산 시작" }).click();
  const eom = await finishedJob(page);
  expect(eom.result.eom.converged).toBe(true);
  await page.getByRole("button", { name: "여기 상태", exact: true }).click();
  await expect(page.getByTestId("eom-states").locator("tbody tr")).toHaveCount(
    1,
  );
  await expect(page.getByTestId("eom-results")).toContainText(
    "oscillator strength·흡수 스펙트럼·NTO와 상태 수명은 계산하지 않았습니다",
  );
  await expect(
    page.getByRole("heading", { name: "흡수 스펙트럼과 상태도" }),
  ).toHaveCount(0);
  const csv = await downloadText(page, "데이터 CSV");
  expect(csv).toContain("energy_ev");
  expect(errors).toEqual([]);
});

test("TDHF: actual singlet roots rendered and method settings remain explicit", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await openWorkbench(page);
  await page.getByLabel("예제 선택").selectOption("formaldehyde_tdhf");
  await expect(page.getByLabel("방법론")).toHaveValue("HF");
  await page.getByRole("button", { name: "계산 시작" }).click();
  const job = await finishedJob(page);
  expect(job.result.excitations).toHaveLength(3);
  await page.getByRole("button", { name: "여기 상태", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "흡수 스펙트럼과 상태도" }),
  ).toBeVisible();
  await expect(page.locator(".result-content tbody tr")).toHaveCount(3);
  await page.getByRole("button", { name: "물성·전하", exact: true }).click();
  await page.setViewportSize({ width: 390, height: 844 });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBe(true);
  expect(errors).toEqual([]);
});

test("comparison diagnostics (UI fixtures): changed model, geometry and unconverged values never receive a delta", async ({
  page,
}) => {
  const source = (await (await page.request.get("/api/examples")).json()).find(
    (example: { id: string }) => example.id === "h2",
  ).spec;
  const fixture = (suffix: string, name: string, energy: number) => ({
    id: `10000000-0000-4000-8000-00000000000${suffix}`,
    status: "succeeded",
    created_at: "2026-11-13T16:00:00Z",
    updated_at: "2026-11-13T16:00:00Z",
    spec: { ...structuredClone(source), name },
    result: {
      converged: true,
      scf_converged: true,
      energy_hartree: energy,
      atoms: structuredClone(source.atoms),
      engine: { name: "UI fixture engine", version: "test" },
      provenance: {
        basis_provider: "UI fixture",
        basis_provider_version: "test",
        numpy_version: "test",
        libxc_version: "test",
      },
    },
  });
  const baseline = fixture("1", "기준 fixture", -1);
  const compatible = fixture("2", "다른 기저 fixture", -1.1);
  compatible.spec.basis = "def2-svp";
  const geometry = fixture("3", "다른 구조 fixture", -1.2);
  geometry.spec.atoms[1].position[2] = 1.2;
  geometry.result.atoms[1].position[2] = 1.2;
  const method = fixture("4", "다른 방법 fixture", -1.3);
  method.spec.method = "DFT";
  const failed = fixture("5", "미수렴 fixture", -1.4);
  failed.status = "unconverged";
  failed.result.converged = false;
  const missingVersion = fixture("6", "버전 누락 fixture", -1.5);
  missingVersion.result.provenance.numpy_version = "";
  const invalidGeometry = fixture("7", "좌표 누락 fixture", -1.6);
  invalidGeometry.result.atoms[0].position = [0, 0];
  const records = [
    baseline,
    compatible,
    geometry,
    method,
    failed,
    missingVersion,
    invalidGeometry,
  ];
  await page.addInitScript((id) => {
    localStorage.setItem("quantum-lab.job", id);
    localStorage.setItem("quantum-lab.view", "workbench");
  }, baseline.id);
  await page.route("**/api/jobs", (route) => route.fulfill({ json: records }));
  await page.route(`**/api/jobs/${baseline.id}`, (route) =>
    route.fulfill({ json: baseline }),
  );
  await page.route(`**/api/jobs/${baseline.id}/history`, (route) =>
    route.fulfill({ json: [] }),
  );
  await openWorkbench(page);
  await expect(page.getByTestId("job-status")).toContainText("수렴 완료");
  await page.getByRole("button", { name: "계산 비교", exact: true }).click();
  const rows = page.getByTestId("job-comparison").locator("tbody tr");
  await expect(rows).toHaveCount(2);
  await expect(
    rows.filter({ hasText: "다른 기저 fixture" }).locator("td").nth(8),
  ).toHaveText("-0.100000000");
  await page.getByLabel("비교 가능한 계산만 보기").uncheck();
  await expect(rows).toHaveCount(7);
  for (const name of [
    "다른 구조 fixture",
    "다른 방법 fixture",
    "미수렴 fixture",
  ]) {
    await expect(
      rows.filter({ hasText: name }).locator("td").nth(8),
    ).toHaveText("—");
  }
});

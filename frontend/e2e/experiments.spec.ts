import { readFile } from "node:fs/promises";
import { expect, test, type Page } from "@playwright/test";

const basisOrder = ["sto-3g", "def2-svp", "def2-tzvp"];

async function openLab(page: Page) {
  await page.goto("/");
  await page.getByRole("button", { name: "학습 실험", exact: true }).click();
  await expect(
    page.getByRole("heading", {
      name: "물 분자의 기저를 바꾸면?",
      exact: true,
    }),
  ).toBeVisible();
}

async function selectedExperiment(page: Page) {
  await expect
    .poll(() =>
      page.evaluate(() => localStorage.getItem("quantum-lab.experiment")),
    )
    .toBeTruthy();
  return (await page.evaluate(() =>
    localStorage.getItem("quantum-lab.experiment"),
  )) as string;
}

async function chartData(page: Page) {
  return page
    .locator(".lab-observations .js-plotly-plot")
    .evaluate((element) => {
      const chart = element as HTMLElement & {
        data: { x: string[]; y: (number | null)[]; connectgaps: boolean }[];
      };
      const { x, y, connectgaps } = chart.data[0];
      return { x, y, connectgaps };
    });
}

test("guided experiment: real three-basis water run, ordered comparison, reload and exports", async ({
  page,
}) => {
  const failures: string[] = [];
  page.on("pageerror", (error) => failures.push(error.message));
  await openLab(page);
  const name = `물 기저 비교 E2E ${Date.now()}`;
  await page.getByLabel("실험 이름", { exact: true }).fill(name);
  await page
    .getByRole("button", { name: "낮아질 것 같아요", exact: true })
    .click();
  await page
    .getByRole("button", { name: "기저 3개 비교 시작", exact: true })
    .click();
  await expect(page.getByTestId("experiment-status")).toContainText(
    "3개 계산 완료",
    {
      timeout: 150000,
    },
  );
  const id = await selectedExperiment(page);
  const response = await page.request.get(`/api/experiments/${id}`);
  expect(response.ok()).toBeTruthy();
  const experiment = await response.json();
  expect(experiment.name).toBe(name);
  expect(experiment.prediction).toBe("lower");
  expect(experiment.status).toBe("completed");
  expect(experiment.rows.map((row: { basis: string }) => row.basis)).toEqual(
    basisOrder,
  );
  expect(experiment.jobs).toHaveLength(3);
  expect(
    new Set(experiment.jobs.map((job: { id: string }) => job.id)).size,
  ).toBe(3);
  expect(experiment.analysis.comparable).toBe(true);

  const commonInputs = experiment.jobs.map(
    (job: { spec: Record<string, unknown> }) => {
      const { name: _name, basis: _basis, ...fixed } = job.spec;
      return fixed;
    },
  );
  expect(commonInputs[1]).toEqual(commonInputs[0]);
  expect(commonInputs[2]).toEqual(commonInputs[0]);
  expect(commonInputs[0]).toMatchObject({
    method: "DFT",
    functional: "PBE",
    grid_level: 3,
    conv_tol: 1e-9,
    threads: 1,
  });

  const tableRows = page.getByTestId("experiment-table").locator("tbody tr");
  await expect(tableRows).toHaveCount(3);
  for (let index = 0; index < 3; index += 1) {
    const row = experiment.rows[index];
    const job = experiment.jobs.find(
      (entry: { id: string }) => entry.id === row.job_id,
    );
    expect(row.status).toBe("succeeded");
    expect(row.comparable).toBe(true);
    expect(row.energy_hartree).toBeCloseTo(job.result.energy_hartree, 10);
    expect(row.basis_functions).toBeGreaterThan(0);
    expect(row.iteration_count).toBeGreaterThan(0);
    expect(row.elapsed_seconds).toBeGreaterThan(0);
    expect(row.peak_memory_mb).toBeGreaterThan(0);
    await expect(tableRows.nth(index)).toContainText(
      basisOrder[index].toUpperCase(),
    );
    await expect(
      tableRows.nth(index).getByTestId("experiment-energy"),
    ).toHaveText(row.energy_hartree.toFixed(9));
    if (index === 0) expect(row.delta_previous_hartree).toBeNull();
    else {
      expect(row.delta_previous_hartree).toBeCloseTo(
        row.energy_hartree - experiment.rows[index - 1].energy_hartree,
        10,
      );
      expect(row.basis_functions).toBeGreaterThan(
        experiment.rows[index - 1].basis_functions,
      );
    }
  }

  for (const [metric, field] of [
    ["energy", "energy_hartree"],
    ["delta", "delta_previous_hartree"],
    ["time", "elapsed_seconds"],
    ["memory", "peak_memory_mb"],
  ]) {
    await page
      .getByLabel("실험 그래프 지표", { exact: true })
      .selectOption(metric);
    await expect
      .poll(async () => (await chartData(page)).y)
      .toEqual(
        experiment.rows.map((row: Record<string, number | null>) => row[field]),
      );
    const chart = await chartData(page);
    expect(chart.x).toEqual(basisOrder.map((basis) => basis.toUpperCase()));
    expect(chart.connectgaps).toBe(false);
  }
  await page
    .getByLabel("실험 그래프 지표", { exact: true })
    .selectOption("energy");

  await page.reload();
  await expect(page.getByTestId("experiment-status")).toContainText(
    "3개 계산 완료",
  );
  expect(await selectedExperiment(page)).toBe(id);
  await expect(page.getByLabel("저장된 실험", { exact: true })).toHaveValue(id);
  for (const format of ["CSV", "JSON"]) {
    const pending = page.waitForEvent("download");
    await page
      .getByRole("link", { name: `실험 ${format} 저장`, exact: true })
      .click();
    const download = await pending;
    expect(download.suggestedFilename()).toContain(`.${format.toLowerCase()}`);
    const path = await download.path();
    expect(path).toBeTruthy();
    const content = await readFile(path!, "utf8");
    basisOrder.forEach((basis) => expect(content).toContain(basis));
    if (format === "JSON") {
      const exported = JSON.parse(content);
      expect(exported.id).toBe(id);
      expect(exported.rows).toEqual(experiment.rows);
      for (const job of experiment.jobs) {
        expect(exported.job_artifacts[job.id]["basis.json"]).toBeTruthy();
        expect(exported.events[job.id].length).toBeGreaterThan(0);
      }
    } else {
      const [header, ...records] = content.trim().split(/\r?\n/);
      const columns = header.split(",");
      expect(records).toHaveLength(3);
      records.forEach((record, index) => {
        const values = record.split(",");
        expect(values[columns.indexOf("basis")]).toBe(basisOrder[index]);
        expect(Number(values[columns.indexOf("energy_hartree")])).toBeCloseTo(
          experiment.rows[index].energy_hartree,
          10,
        );
        if (index === 0)
          expect(values[columns.indexOf("delta_previous_hartree")]).toBe("");
      });
    }
  }
  await page.screenshot({
    path: "test-results/experiment-workspace.png",
    fullPage: true,
  });
  await tableRows
    .nth(1)
    .getByRole("button", { name: "계산 자세히 보기", exact: true })
    .click();
  await expect(page.getByTestId("job-status")).toContainText("수렴 완료");
  expect(
    await page.evaluate(() => localStorage.getItem("quantum-lab.job")),
  ).toBe(experiment.rows[1].job_id);
  await expect(page.getByLabel("기저 집합", { exact: true })).toHaveValue(
    "def2-svp",
  );
  await page.getByRole("button", { name: "학습 실험", exact: true }).click();
  await expect(page.getByTestId("experiment-status")).toContainText(
    "3개 계산 완료",
  );
  await page.getByRole("button", { name: "새 실험 준비", exact: true }).click();
  await expect(page.getByTestId("experiment-table")).toHaveCount(0);
  await expect(page.getByTestId("experiment-status")).toHaveCount(0);
  await expect(page.getByLabel("실험 이름", { exact: true })).toHaveValue(
    "물의 기저 비교",
  );
  await page.getByLabel("저장된 실험", { exact: true }).selectOption(id);
  await expect(page.getByTestId("experiment-status")).toContainText(
    "3개 계산 완료",
  );
  await expect(
    page.getByTestId("experiment-table").locator("tbody tr"),
  ).toHaveCount(3);
  expect(failures).toEqual([]);
});

test("guided experiment: lost creation response retries the persisted request without duplicate jobs", async ({
  page,
}) => {
  let lost = false;
  let originalId = "";
  const requests: Record<string, unknown>[] = [];
  await page.route("**/api/experiments", async (route) => {
    if (route.request().method() !== "POST") {
      await route.continue();
      return;
    }
    requests.push(route.request().postDataJSON());
    if (!lost) {
      lost = true;
      // Execute the real request, then simulate losing its response in transit.
      const response = await route.fetch();
      expect(response.ok()).toBeTruthy();
      originalId = (await response.json()).id;
      await route.abort("failed");
    } else await route.continue();
  });
  await openLab(page);
  await page
    .getByLabel("실험 이름", { exact: true })
    .fill(`응답 복구 E2E ${Date.now()}`);
  await page
    .getByRole("button", { name: "기저 3개 비교 시작", exact: true })
    .click();
  await expect(
    page.getByRole("button", { name: "같은 요청 다시 보내기", exact: true }),
  ).toBeVisible();
  expect(originalId).toBeTruthy();
  await page.reload();
  await page
    .getByRole("button", { name: "같은 요청 다시 보내기", exact: true })
    .click();
  await expect(page.getByTestId("experiment-status")).toContainText(
    "3개 계산 완료",
    {
      timeout: 150000,
    },
  );
  expect(await selectedExperiment(page)).toBe(originalId);
  expect(requests).toHaveLength(2);
  expect(requests[1]).toEqual(requests[0]);
  const list = await (await page.request.get("/api/experiments")).json();
  expect(
    list.filter(
      (entry: { request_id: string }) =>
        entry.request_id === requests[0].request_id,
    ),
  ).toHaveLength(1);
  const detail = await (
    await page.request.get(`/api/experiments/${originalId}`)
  ).json();
  expect(detail.jobs).toHaveLength(3);
  expect(
    await page.evaluate(() =>
      localStorage.getItem("quantum-lab.experiment.pending"),
    ),
  ).toBeNull();
});

test("guided experiment: group cancellation keeps terminal states and survives reload", async ({
  page,
}) => {
  await openLab(page);
  await page
    .getByLabel("실험 이름", { exact: true })
    .fill(`취소 E2E ${Date.now()}`);
  await page
    .getByRole("button", { name: "기저 3개 비교 시작", exact: true })
    .click();
  await page.getByRole("button", { name: "실험 취소", exact: true }).click();
  const id = await selectedExperiment(page);
  await expect
    .poll(
      async () =>
        (await (await page.request.get(`/api/experiments/${id}`)).json())
          .status,
    )
    .toBe("cancelled");
  const detail = await (
    await page.request.get(`/api/experiments/${id}`)
  ).json();
  expect(
    detail.rows.some((row: { status: string }) => row.status === "cancelled"),
  ).toBe(true);
  expect(
    detail.rows.every(
      (row: { status: string }) => !["queued", "running"].includes(row.status),
    ),
  ).toBe(true);
  expect(detail.analysis.complete).toBe(true);
  expect(detail.analysis.comparable).toBe(false);
  for (let index = 1; index < detail.rows.length; index += 1) {
    if (!detail.rows[index].comparable || !detail.rows[index - 1].comparable)
      expect(detail.rows[index].delta_previous_hartree).toBeNull();
  }
  await page.reload();
  await expect(page.getByTestId("experiment-status")).toContainText("취소");
  expect(await selectedExperiment(page)).toBe(id);
  await page
    .getByLabel("실험 그래프 지표", { exact: true })
    .selectOption("delta");
  await expect
    .poll(async () => (await chartData(page)).y)
    .toEqual(
      detail.rows.map(
        (row: { delta_previous_hartree: number | null }) =>
          row.delta_previous_hartree,
      ),
    );
  expect((await chartData(page)).connectgaps).toBe(false);
  await expect(
    page.getByRole("button", { name: "새 실험 준비", exact: true }),
  ).toBeVisible();
});

test("guided experiment: mobile instructions fit without editing structure JSON", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await openLab(page);
  await expect(
    page.getByRole("button", { name: "기저 3개 비교 시작", exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "아직 모르겠어요", exact: true }),
  ).toBeVisible();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  await page.screenshot({
    path: "test-results/experiment-mobile.png",
    fullPage: true,
  });
});

test("guided experiment: failed middle result leaves gaps and switching clears stale data (UI fixtures)", async ({
  page,
}) => {
  // These marked responses test display semantics only, not scientific accuracy.
  const templateResponse = await page.request.get("/api/experiments/templates");
  expect(templateResponse.ok()).toBe(true);
  const [template] = await templateResponse.json();
  const failed = {
    id: "ui-fixture-failed",
    template_id: "water_basis_v1",
    name: "실패 표시 UI 테스트 (가상 응답)",
    prediction: "unsure",
    request_id: "00000000-0000-4000-8000-000000000001",
    created_at: new Date().toISOString(),
    cancel_requested: false,
    base_spec: template.fixed_spec,
    bases: basisOrder,
    status: "completed_with_issues",
    completed_jobs: 3,
    successful_jobs: 2,
    total_jobs: 3,
    active_job_id: null,
    rows: basisOrder.map((basis, index) => ({
      ordinal: index,
      job_id: `ui-fixture-job-${index}`,
      basis,
      status: index === 1 ? "failed" : "succeeded",
      comparable: index !== 1,
      energy_hartree: index === 0 ? -75 : index === 2 ? -76 : null,
      delta_previous_hartree: null,
      basis_functions: null,
      iteration_count: null,
      elapsed_seconds: null,
      peak_memory_mb: null,
      error: index === 1 ? "SCF 반복 한도 도달 (UI fixture)" : null,
      comparison_note:
        index === 1
          ? "최종 SCF 수렴을 확인하지 못했습니다. (UI fixture)"
          : null,
    })),
    jobs: [],
    analysis: {
      valid_count: 2,
      total_count: 3,
      complete: true,
      comparable: false,
      energy_trend: null,
      last_step_delta_hartree: null,
      total_change_hartree: null,
      messages: [
        { kind: "limit", text: "가상 응답으로 실패 표시만 검사합니다." },
      ],
    },
  };
  const queued = {
    ...failed,
    id: "ui-fixture-queued",
    name: "새 상태 UI 테스트 (가상 응답)",
    status: "queued",
    completed_jobs: 0,
    successful_jobs: 0,
    rows: failed.rows.map((row) => ({
      ...row,
      status: "queued",
      comparable: false,
      energy_hartree: null,
      error: null,
    })),
    analysis: { ...failed.analysis, valid_count: 0, complete: false },
  };
  let releaseNext = () => {};
  const nextResponse = new Promise<void>((resolve) => {
    releaseNext = resolve;
  });
  await page.route("**/api/experiments", (route) =>
    route.fulfill({ json: [failed, queued] }),
  );
  await page.route(`**/api/experiments/${failed.id}`, (route) =>
    route.fulfill({ json: failed }),
  );
  await page.route(`**/api/experiments/${queued.id}`, async (route) => {
    await nextResponse;
    await route.fulfill({ json: queued });
  });
  await page.addInitScript((id) => {
    localStorage.setItem("quantum-lab.experiment", id);
    localStorage.setItem("quantum-lab.view", "lab");
  }, failed.id);
  await openLab(page);
  await expect(page.getByTestId("experiment-status")).toContainText(
    "일부 계산에 진단 필요",
  );
  const rows = page.getByTestId("experiment-table").locator("tbody tr");
  await expect(rows.nth(1).getByTestId("experiment-energy")).toHaveText("—");
  await rows.nth(1).getByText("진단", { exact: true }).click();
  await expect(rows.nth(1)).toContainText("SCF 반복 한도 도달 (UI fixture)");
  await rows.nth(1).getByText("비교 제외 사유", { exact: true }).click();
  await expect(rows.nth(1)).toContainText(
    "최종 SCF 수렴을 확인하지 못했습니다. (UI fixture)",
  );
  await expect
    .poll(async () => (await chartData(page)).y)
    .toEqual([-75, null, -76]);
  expect((await chartData(page)).connectgaps).toBe(false);
  await page
    .getByLabel("실험 그래프 지표", { exact: true })
    .selectOption("delta");
  await expect
    .poll(async () => (await chartData(page)).y)
    .toEqual([null, null, null]);
  await expect(
    page.getByText("가상 응답으로 실패 표시만 검사합니다.", { exact: true }),
  ).toBeVisible();
  try {
    await page
      .getByLabel("저장된 실험", { exact: true })
      .selectOption(queued.id);
    await expect(page.getByTestId("experiment-table")).toHaveCount(0);
    await expect(page.getByTestId("experiment-status")).toHaveCount(0);
    await expect(
      page.getByText("가상 응답으로 실패 표시만 검사합니다.", { exact: true }),
    ).toHaveCount(0);
  } finally {
    releaseNext();
  }
  await expect(page.getByTestId("experiment-status")).toContainText(
    "실험 대기 중",
  );
  await expect
    .poll(async () => (await chartData(page)).y)
    .toEqual([null, null, null]);
});

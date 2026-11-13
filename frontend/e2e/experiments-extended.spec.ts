import { readFile } from "node:fs/promises";
import { expect, test, type Page } from "@playwright/test";

async function selectTemplate(page: Page, id: string) {
  await page.goto("/");
  await page.getByRole("button", { name: "학습 실험", exact: true }).click();
  await page.getByLabel("학습 실험 주제", { exact: true }).selectOption(id);
}
async function runExperiment(page: Page, count: number) {
  await page
    .getByRole("button", { name: `${count}개 조건 비교 시작`, exact: true })
    .click();
  await expect(page.getByTestId("experiment-status")).toContainText(
    `${count}개 계산 완료`,
    { timeout: 170000 },
  );
  const id = await page.evaluate(() =>
    localStorage.getItem("quantum-lab.experiment"),
  );
  expect(id).toBeTruthy();
  const response = await page.request.get(`/api/experiments/${id}`);
  expect(response.ok()).toBe(true);
  return response.json();
}
async function plotValues(page: Page) {
  return page
    .locator(".lab-observations .js-plotly-plot")
    .evaluate((element) => {
      const chart = element as HTMLElement & {
        data: {
          x: (number | string)[];
          y: (number | null)[];
          connectgaps: boolean;
        }[];
      };
      return {
        x: chart.data[0].x,
        y: chart.data[0].y,
        connectgaps: chart.data[0].connectgaps,
      };
    });
}

test("extended experiments: actual integration-grid comparison and frozen historical conditions", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await selectTemplate(page, "water_grid_v1");
  await expect(
    page.getByRole("heading", {
      name: "물의 적분 격자를 촘촘하게 하면?",
      exact: true,
    }),
  ).toBeVisible();
  await expect(page.locator(".lab-variable")).toContainText("적분 격자");
  const experiment = await runExperiment(page, 3);
  expect(experiment.definition.variable).toBe("grid_level");
  expect(experiment.definition.values).toEqual([1, 3, 5]);
  expect(
    experiment.jobs.map(
      (job: { spec: { grid_level: number } }) => job.spec.grid_level,
    ),
  ).toEqual([1, 3, 5]);
  expect(
    new Set(
      experiment.jobs.map((job: { spec: { basis: string } }) => job.spec.basis),
    ).size,
  ).toBe(1);
  await expect
    .poll(async () => (await plotValues(page)).y)
    .toEqual(
      experiment.rows.map(
        (row: { energy_hartree: number }) => row.energy_hartree,
      ),
    );
  expect((await plotValues(page)).x).toEqual(
    experiment.rows.map((row: { label: string }) => row.label),
  );
  const originalId = experiment.id;
  await page.reload();
  await expect(page.getByTestId("experiment-status")).toContainText(
    "3개 계산 완료",
  );
  await expect(page.getByLabel("학습 실험 주제", { exact: true })).toHaveValue(
    "water_grid_v1",
  );
  await page
    .getByLabel("학습 실험 주제", { exact: true })
    .selectOption("water_functional_v1");
  await expect(page.getByTestId("experiment-table")).toHaveCount(0);
  await expect(page.locator(".lab-variable")).toContainText("PBE0");
  await page
    .getByLabel("저장된 실험", { exact: true })
    .selectOption(originalId);
  await expect(page.getByLabel("학습 실험 주제", { exact: true })).toHaveValue(
    "water_grid_v1",
  );
  await expect(page.getByTestId("experiment-status")).toContainText(
    "3개 계산 완료",
  );
  await page.screenshot({
    path: "test-results/experiment-grid.png",
    fullPage: true,
  });
  expect(errors).toEqual([]);
});

test("extended experiments: actual H2 scan connects numeric samples to structures without claiming dynamics", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await selectTemplate(page, "hydrogen_dissociation_v1");
  const experiment = await runExperiment(page, 5);
  const distances = [0.5, 0.74, 1, 1.5, 2];
  expect(experiment.definition.values).toEqual(distances);
  expect(experiment.rows.map((row: { value: number }) => row.value)).toEqual(
    distances,
  );
  await expect.poll(async () => (await plotValues(page)).x).toEqual(distances);
  expect((await plotValues(page)).y).toEqual(
    experiment.rows.map(
      (row: { energy_hartree: number }) => row.energy_hartree,
    ),
  );
  expect((await plotValues(page)).connectgaps).toBe(false);
  const structure = page.getByLabel("표시할 계산 구조", { exact: true });
  await structure.selectOption(experiment.rows[4].job_id);
  await expect(page.getByTestId("scan-structure-caption")).toContainText(
    experiment.rows[4].label,
  );
  await expect(page.getByTestId("scan-structure-caption")).toContainText(
    "실제 시간 궤적이 아닙니다",
  );
  await page.locator(".lab-coordinate-details summary").click();
  await expect(
    page.locator(".lab-coordinate-details tbody tr").nth(1),
  ).toContainText("2.000");
  await page
    .getByRole("button", { name: "구조 순서대로 보기", exact: true })
    .click();
  await expect
    .poll(() => structure.inputValue(), { timeout: 5000 })
    .not.toBe(experiment.rows[4].job_id);
  await page
    .getByRole("button", { name: "구조 순서 보기 멈춤", exact: true })
    .click();
  const downloadPromise = page.waitForEvent("download");
  await page.getByRole("link", { name: "실험 JSON 저장", exact: true }).click();
  const download = await downloadPromise;
  const exported = JSON.parse(await readFile((await download.path())!, "utf8"));
  expect(exported.definition.variable).toBe("bond_distance");
  expect(exported.definition.expected_specs).toHaveLength(5);
  expect(exported.rows).toEqual(experiment.rows);
  await page.screenshot({
    path: "test-results/experiment-scan.png",
    fullPage: true,
  });
  expect(errors).toEqual([]);
});

test("extended experiments: actual crystal FFT comparison preserves cell and per-atom energies", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await selectTemplate(page, "diamond_mesh_v1");
  const experiment = await runExperiment(page, 3);
  expect(experiment.domain).toBe("crystal");
  expect(
    experiment.jobs.map((job: { spec: { mesh: number[] } }) => job.spec.mesh),
  ).toEqual([
    [15, 15, 15],
    [21, 21, 21],
    [27, 27, 27],
  ]);
  for (const row of experiment.rows) {
    expect(row.comparable).toBe(true);
    expect(row.energy_per_atom_hartree).toBeCloseTo(row.energy_hartree / 2, 11);
  }
  await page
    .getByLabel("실험 그래프 지표", { exact: true })
    .selectOption("energy_per_atom");
  await expect
    .poll(async () => (await plotValues(page)).y)
    .toEqual(
      experiment.rows.map(
        (row: { energy_per_atom_hartree: number }) =>
          row.energy_per_atom_hartree,
      ),
    );
  await expect(page.getByTestId("experiment-table")).toContainText(
    "셀 에너지 / Eh",
  );
  await expect(page.getByTestId("experiment-table")).toContainText(
    "원자당 / Eh",
  );
  await page
    .getByLabel("실험 그래프 지표", { exact: true })
    .selectOption("delta_per_atom");
  await expect
    .poll(async () => (await plotValues(page)).y)
    .toEqual(
      experiment.rows.map(
        (row: { delta_previous_per_atom_hartree: number | null }) =>
          row.delta_previous_per_atom_hartree,
      ),
    );
  await page.screenshot({
    path: "test-results/experiment-crystal.png",
    fullPage: true,
  });
  expect(errors).toEqual([]);
});

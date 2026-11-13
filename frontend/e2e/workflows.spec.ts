import { test, expect } from "@playwright/test";

test("molecule: actual H2 calculation, persisted reload, CSV export and density cache", async ({
  page,
}) => {
  const failures: string[] = [];
  page.on("pageerror", (e) => failures.push(e.message));
  await page.goto("/");
  await expect(page.getByText("로컬 엔진 연결됨")).toBeVisible();
  const options = await page
    .getByLabel("예제 선택")
    .locator("option")
    .evaluateAll((nodes) =>
      nodes.map((n) => ({
        value: (n as HTMLOptionElement).value,
        text: n.textContent ?? "",
      })),
    );
  const hydrogen = options.find(
    (o) => o.value === "h2" || o.text.includes("H₂") || o.text.includes("수소"),
  );
  expect(hydrogen).toBeTruthy();
  await page.getByLabel("예제 선택").selectOption(hydrogen!.value);
  await page
    .locator("input[type=file]")
    .setInputFiles({
      name: "h2.xyz",
      mimeType: "chemical/x-xyz",
      buffer: Buffer.from("2\nHydrogen E2E\nH 0 0 0\nH 0 0 0.74\n"),
    });
  await expect(page.getByLabel("계산 이름")).toHaveValue("h2.xyz");
  await page.getByRole("button", { name: "계산 시작" }).click();
  await expect(page.getByTestId("job-status")).toContainText("수렴 완료", {
    timeout: 120000,
  });
  await expect(page.getByTestId("energy")).toContainText("-1.");
  await expect(page.locator(".molecule-viewer canvas")).toBeVisible();
  await expect(page.locator(".viewer-error")).toHaveCount(0);
  const jobId = await page.evaluate(() =>
    localStorage.getItem("quantum-lab.job"),
  );
  expect(jobId).toBeTruthy();
  await page.reload();
  await expect(page.getByTestId("job-status")).toContainText("수렴 완료");
  expect(
    await page.evaluate(() => localStorage.getItem("quantum-lab.job")),
  ).toBe(jobId);
  await expect(page.locator(".convergence-panel .count-pill")).not.toHaveText(
    "0 STEPS",
  );
  const download = page.waitForEvent("download");
  await page.getByRole("link", { name: "반복 CSV" }).click();
  expect((await download).suggestedFilename()).toContain(".csv");
  await page.getByRole("button", { name: "격자 생성", exact: true }).click();
  await expect(page.getByRole("link", { name: "CUBE 저장" })).toBeVisible({
    timeout: 120000,
  });
  await expect(page.locator(".viewer-error")).toHaveCount(0);
  await page.getByRole("button", { name: "숨기기", exact: true }).click();
  await page.getByRole("button", { name: "격자 생성", exact: true }).click();
  await expect(page.getByRole("link", { name: "CUBE 저장" })).toBeVisible({
    timeout: 30000,
  });
  await page.getByRole("button", { name: "전자 구조", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "전자 구조", exact: true }),
  ).toBeVisible();
  await page.screenshot({
    path: "test-results/molecule-workspace.png",
    fullPage: true,
  });
  expect(failures).toEqual([]);
});

test("crystal: CIF upload, actual diamond bands, unit cell and JSON export", async ({
  page,
}) => {
  const failures: string[] = [];
  page.on("pageerror", (e) => failures.push(e.message));
  await page.goto("/");
  await expect(page.getByText("로컬 엔진 연결됨")).toBeVisible();
  const cif = `data_diamond\n_cell_length_a 2.522108468\n_cell_length_b 2.522108468\n_cell_length_c 2.522108468\n_cell_angle_alpha 60\n_cell_angle_beta 60\n_cell_angle_gamma 60\n_space_group_name_H-M_alt 'P 1'\nloop_\n_atom_site_label\n_atom_site_type_symbol\n_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\nC1 C 0 0 0\nC2 C 0.25 0.25 0.25\n`;
  await page.locator("input[type=file]").setInputFiles({
    name: "diamond.cif",
    mimeType: "chemical/x-cif",
    buffer: Buffer.from(cif),
  });
  await expect(page.getByLabel("계산 작업")).toHaveValue("bands");
  await page.getByRole("button", { name: "계산 시작" }).click();
  await expect(page.getByTestId("job-status")).toContainText("수렴 완료", {
    timeout: 150000,
  });
  await page.getByRole("button", { name: "밴드 구조", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "밴드 구조", exact: true }),
  ).toBeVisible();
  await expect(page.getByText(/표본 KS gap/)).toBeVisible();
  await page.getByLabel("단위격자 반복").selectOption("2");
  await expect(page.locator(".viewer-error")).toHaveCount(0);
  const download = page.waitForEvent("download");
  await page.getByRole("link", { name: "결과 저장" }).click();
  expect((await download).suggestedFilename()).toContain(".json");
  await page.reload();
  await expect(page.getByTestId("job-status")).toContainText("수렴 완료");
  await page.getByRole("button", { name: "밴드 구조", exact: true }).click();
  await page.screenshot({
    path: "test-results/crystal-workspace.png",
    fullPage: true,
  });
  expect(failures).toEqual([]);
});

test("mobile layout and failed input show an actionable diagnosis", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await expect(page.getByText("로컬 엔진 연결됨")).toBeAttached();
  await page.getByLabel("원자 좌표").fill("H not-a-number 0 0");
  await page.getByRole("button", { name: "계산 시작" }).click();
  await expect(page.getByRole("alert")).toContainText("원소 x y z");
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBeTruthy();
});

test("excited states: actual formaldehyde TDA, spectrum, conceptual transition and NTO", async ({
  page,
}) => {
  const failures: string[] = [];
  page.on("pageerror", (e) => failures.push(e.message));
  await page.goto("/");
  await expect(page.getByText("로컬 엔진 연결됨")).toBeVisible();
  await page.getByLabel("예제 선택").selectOption("formaldehyde");
  await page.getByRole("button", { name: "계산 시작" }).click();
  await expect(page.getByTestId("job-status")).toContainText("수렴 완료", {
    timeout: 150000,
  });
  await page.getByRole("button", { name: "여기 상태", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "흡수 스펙트럼과 상태도" }),
  ).toBeVisible();
  await expect(page.locator(".result-content tbody tr")).not.toHaveCount(0);
  await page.getByRole("button", { name: "전이 개념 보기" }).click();
  await expect(page.getByText("교육용 모식도 · 시간 궤적 아님")).toBeVisible();
  await expect(page.locator(".state-diagram")).toHaveClass(/animating/);
  await page.getByLabel("3D 표시 종류").selectOption("nto_hole");
  await page.getByRole("button", { name: "격자 생성", exact: true }).click();
  await expect(page.getByRole("link", { name: "CUBE 저장" })).toBeVisible({
    timeout: 120000,
  });
  await expect(page.locator(".viewer-error")).toHaveCount(0);
  await page.screenshot({
    path: "test-results/excited-workspace.png",
    fullPage: true,
  });
  expect(failures).toEqual([]);
});

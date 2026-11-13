import { expect, test } from "@playwright/test";

test("learner start: H2 preparation is explicit, first-screen run and explanations preserve the calculation", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1440, height: 768 });
  const errors: string[] = [];
  const submissions: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("request", (request) => {
    if (request.method() === "POST" && request.url().endsWith("/api/jobs"))
      submissions.push(request.url());
  });
  await page.goto("/");
  const load = page.getByRole("button", {
    name: "H₂ 예제 불러오기",
    exact: true,
  });
  await expect(load).toBeEnabled();
  await load.click();
  await expect(page.getByLabel("전하", { exact: true })).toHaveValue("0");
  await expect(page.getByLabel("스핀", { exact: true })).toHaveValue("0");
  await expect(page.getByLabel("방법론", { exact: true })).toHaveValue("HF");
  await expect(page.getByLabel("기저 집합", { exact: true })).toHaveValue(
    "sto-3g",
  );
  const coordinates = await page
    .getByLabel("원자 좌표", { exact: true })
    .inputValue();
  expect(coordinates.trim().split("\n")).toHaveLength(2);
  expect(
    coordinates
      .trim()
      .split("\n")
      .every((row) => row.startsWith("H ")),
  ).toBe(true);
  await expect(page.getByTestId("job-status")).toHaveText("계산 준비");
  expect(submissions).toHaveLength(0);
  expect(
    await page.evaluate(() => localStorage.getItem("quantum-lab.job")),
  ).toBeNull();

  const run = page.getByRole("button", { name: "계산 시작", exact: true });
  await expect(run).toHaveCount(1);
  const position = await run.boundingBox();
  expect(position).not.toBeNull();
  expect(position!.y).toBeGreaterThanOrEqual(0);
  expect(position!.y + position!.height).toBeLessThanOrEqual(768);
  await expect(page.getByLabel("기저 집합", { exact: true })).toHaveAttribute(
    "aria-describedby",
    "basis-help",
  );
  await expect(page.locator("#basis-help")).toContainText("수학 함수");
  await run.click();
  await expect(page.getByTestId("job-status")).toContainText("수렴 완료", {
    timeout: 120000,
  });
  expect(submissions).toHaveLength(1);
  const jobId = await page.evaluate(() =>
    localStorage.getItem("quantum-lab.job"),
  );
  const response = await page.request.get(`/api/jobs/${jobId}`);
  expect(response.ok()).toBeTruthy();
  const before = await response.json();
  expect(
    before.spec.atoms.map((atom: { symbol: string }) => atom.symbol),
  ).toEqual(["H", "H"]);
  expect(before.result.energy_hartree).toBeCloseTo(-1.116759307507584, 8);
  await expect(page.getByTestId("learner-convergence-note")).toContainText(
    "실험값과 얼마나 맞는지",
  );
  await expect(
    page.getByText("Eh는 에너지 단위 하트리입니다.", { exact: true }),
  ).toBeVisible();

  await page.getByRole("button", { name: "전문가", exact: true }).click();
  await expect(
    page.getByRole("button", { name: "계산 시작", exact: true }),
  ).toHaveCount(1);
  await expect(page.getByLabel("SCF 최대 반복", { exact: true })).toBeVisible();
  await expect(page.getByTestId("learner-convergence-note")).toHaveCount(0);
  const after = await (await page.request.get(`/api/jobs/${jobId}`)).json();
  expect(after.spec).toEqual(before.spec);
  expect(after.result.energy_hartree).toBe(before.result.energy_hartree);
  expect(
    await page.evaluate(() => localStorage.getItem("quantum-lab.job")),
  ).toBe(jobId);
  expect(submissions).toHaveLength(1);
  expect(errors).toEqual([]);
});

import { defineConfig } from "@playwright/test";
import { existsSync } from "node:fs";
export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false,
  workers: 1,
  timeout: 180000,
  expect: { timeout: 30000 },
  reporter: [["list"]],
  use: {
    baseURL: process.env.QUANTUM_LAB_URL ?? "http://127.0.0.1:8000",
    viewport: { width: 1440, height: 1100 },
    screenshot: "only-on-failure",
    trace: "retain-on-failure",
    launchOptions: {
      executablePath:
        process.env.CHROMIUM_PATH ??
        (existsSync("/usr/bin/chromium") ? "/usr/bin/chromium" : undefined),
      args: ["--no-sandbox", "--enable-unsafe-swiftshader"],
    },
  },
});

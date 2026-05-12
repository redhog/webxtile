import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './tests',
  use: { headless: true, browserName: 'chromium' },
  timeout: 30_000,
});

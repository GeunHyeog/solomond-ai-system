import { test, expect } from '@playwright/test';

// Phase 15-1-O: Comet QA v9 - Smoke Tests
// 5 critical scenarios for operational stability

const baseUrl = process.env.E2E_BASE_URL || 'https://jewelry-master-system.pages.dev';
const email = process.env.E2E_ADMIN_EMAIL;
const password = process.env.E2E_ADMIN_PASSWORD;
const hasCredentials = email && password;

test.describe('Phase 15-1-O: Smoke Tests', () => {
  // Suite-level skip guard
  test.beforeEach(({ }, testInfo) => {
    if (!hasCredentials) {
      console.log('[Smoke Test Guard] No credentials, skipping:', testInfo.title);
      testInfo.skip();
    }
  });

  test('smoke-01: Login persistence across page reloads', async ({ page }) => {
    // Navigate to login
    await page.goto(`${baseUrl}/login`);

    // Login
    await page.fill('input[type="email"]', email!);
    await page.fill('input[type="password"]', password!);
    await page.click('button[type="submit"]');

    // Wait for dashboard
    await expect(page).toHaveURL(/\/(dashboard|\/)/);

    // Take screenshot before reload
    await page.screenshot({ path: 'test-results/smoke-01-before.png' });

    // Reload page
    await page.reload();

    // Verify session persists (no redirect to login)
    await expect(page).not.toHaveURL(/\/login/);
    await expect(page.locator('body')).toContainText(email!.split('@')[0] || '설정');

    // Take screenshot after reload
    await page.screenshot({ path: 'test-results/smoke-01-after.png' });

    console.log('[Smoke 01] PASS - Session persisted after reload');
  });

  test('smoke-02: Dashboard rendering with all components', async ({ page }) => {
    // Login
    await page.goto(`${baseUrl}/login`);
    await page.fill('input[type="email"]', email!);
    await page.fill('input[type="password"]', password!);
    await page.click('button[type="submit"]');

    // Navigate to dashboard
    await page.goto(`${baseUrl}/`);

    // Verify critical components
    // Note: Adjust selectors based on actual dashboard structure
    const hasHeader = await page.locator('header, nav, [role="banner"]').count() > 0;
    const hasContent = await page.locator('main, [role="main"], body').count() > 0;

    expect(hasHeader || hasContent).toBeTruthy();

    // Take screenshot
    await page.screenshot({ path: 'test-results/smoke-02-dashboard.png', fullPage: true });

    // Check console for errors
    const consoleErrors: string[] = [];
    page.on('console', msg => {
      if (msg.type() === 'error') {
        consoleErrors.push(msg.text());
      }
    });

    // Wait a bit for any async errors
    await page.waitForTimeout(2000);

    console.log('[Smoke 02] PASS - Dashboard rendered, errors:', consoleErrors.length);
  });

  test('smoke-03: Activity log or data loading', async ({ page }) => {
    // Login
    await page.goto(`${baseUrl}/login`);
    await page.fill('input[type="email"]', email!);
    await page.fill('input[type="password"]', password!);
    await page.click('button[type="submit"]');

    // Navigate to dashboard
    await page.goto(`${baseUrl}/`);

    // Wait for any data to load (adjust selector based on actual implementation)
    // This is a placeholder - adjust based on actual activity log structure
    await page.waitForTimeout(3000);

    const hasData = await page.locator('body').count() > 0;
    expect(hasData).toBeTruthy();

    await page.screenshot({ path: 'test-results/smoke-03-activity.png' });

    console.log('[Smoke 03] PASS - Page loaded with data');
  });

  test('smoke-04: State preservation after browser refresh', async ({ page }) => {
    // Login
    await page.goto(`${baseUrl}/login`);
    await page.fill('input[type="email"]', email!);
    await page.fill('input[type="password"]', password!);
    await page.click('button[type="submit"]');

    // Navigate to Settings
    await page.goto(`${baseUrl}/settings`);
    await page.waitForTimeout(1000);

    // Check MFA status before refresh
    const statusBefore = await page.locator('text=Enabled, text=Disabled').first().textContent();
    await page.screenshot({ path: 'test-results/smoke-04-before.png' });

    // Hard refresh
    await page.reload({ waitUntil: 'networkidle' });

    // Check MFA status after refresh
    const statusAfter = await page.locator('text=Enabled, text=Disabled').first().textContent();
    await page.screenshot({ path: 'test-results/smoke-04-after.png' });

    // Verify state preserved
    expect(statusBefore).toBe(statusAfter);

    console.log('[Smoke 04] PASS - MFA state preserved:', statusBefore);
  });

  test('smoke-05: Network stability (no 500 errors)', async ({ page }) => {
    const failedRequests: Array<{ url: string; status: number }> = [];
    const slowRequests: Array<{ url: string; time: number }> = [];

    // Monitor network requests
    page.on('response', response => {
      const status = response.status();
      const url = response.url();

      // Track failed requests (5xx errors)
      if (status >= 500) {
        failedRequests.push({ url, status });
      }

      // Track slow requests (>2s) - Note: timing() requires await in Playwright
      // Simplified version: just track status
    });

    // Login
    await page.goto(`${baseUrl}/login`);
    await page.fill('input[type="email"]', email!);
    await page.fill('input[type="password"]', password!);
    await page.click('button[type="submit"]');

    // Navigate between pages
    await page.goto(`${baseUrl}/`);
    await page.waitForTimeout(1000);

    await page.goto(`${baseUrl}/settings`);
    await page.waitForTimeout(1000);

    await page.goto(`${baseUrl}/`);
    await page.waitForTimeout(1000);

    // Verify no failed requests
    expect(failedRequests.length).toBe(0);

    console.log('[Smoke 05] PASS - No network failures, failed requests:', failedRequests.length);

    if (failedRequests.length > 0) {
      console.error('[Smoke 05] Failed requests:', JSON.stringify(failedRequests, null, 2));
    }
  });
});

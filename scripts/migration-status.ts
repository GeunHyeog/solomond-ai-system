// scripts/migration-status.ts
import { execSync } from 'child_process';
import { writeFileSync, mkdirSync } from 'fs';
import { join } from 'path';

const PROJECT_REF = process.env.SUPABASE_PROJECT_REF || '';
if (!PROJECT_REF) {
  console.error('❌ SUPABASE_PROJECT_REF missing');
  process.exit(1);
}

const REPORT = join(process.cwd(), 'docs', 'PHASE15_2_MIGRATION_STATUS.md');

function run(cmd: string) {
  return execSync(cmd, { encoding: 'utf-8', stdio: 'pipe' });
}

try {
  const out = run(`npx supabase migration list --project-ref ${PROJECT_REF}`);
  const lines = out.trim().split('\n');

  const applied: string[] = [];
  const pending: string[] = [];
  for (const line of lines) {
    if (/Applied/i.test(line)) applied.push(line);
    else if (/Unapplied|Local/i.test(line)) pending.push(line);
  }

  const md = `# Migration Status\n\n**Generated**: ${new Date().toISOString()}\n**Project**: ${PROJECT_REF}\n\n## ✅ Applied (${applied.length})\n${applied.map(l=>`- ${l}`).join('\n') || '(none)'}\n\n## ⚠️ Pending (${pending.length})\n${pending.map(l=>`- ${l}`).join('\n') || '(none)'}\n\n## Summary\n- Total: ${applied.length + pending.length}\n- Applied: ${applied.length}\n- Pending: ${pending.length}\n- Status: ${pending.length === 0 ? '✅ All synced' : '⚠️ Pending migrations detected'}\n`;
  mkdirSync(join(process.cwd(), 'docs'), { recursive: true });
  writeFileSync(REPORT, md);
  console.log(`✅ Status report written: ${REPORT}`);

  if (process.argv.includes('--ci') && pending.length > 0) {
    console.log('ℹ️ CI mode: pending exists → warn only');
  }
  process.exit(0);
} catch (e: any) {
  console.error('❌ migration-status failed:', e.message);
  process.exit(1);
}

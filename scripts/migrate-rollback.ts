// scripts/migrate-rollback.ts
// NOTE: 휴리스틱 역연산(안전 우선). 복잡한 DDL은 수동 보완 필요.
import { readdirSync, readFileSync, writeFileSync, mkdirSync } from 'fs';
import { join } from 'path';

const MIGR_DIR = join(process.cwd(), 'supabase', 'migrations');
const OUT = join(MIGR_DIR, '.rollback.sql');

function extractRollback(sql: string) {
  const lines = sql.split(/\r?\n/);
  const rollback: string[] = [];

  // 대표 패턴들
  const addColumn = /ALTER\s+TABLE\s+(\S+)\s+ADD\s+COLUMN\s+(\S+)/i;
  const createTable = /CREATE\s+TABLE\s+(IF\s+NOT\s+EXISTS\s+)?(\S+)/i;
  const createIndex = /CREATE\s+(UNIQUE\s+)?INDEX\s+(\S+)\s+ON\s+(\S+)/i;
  const createFn = /CREATE\s+FUNCTION\s+(\S+)/i;
  const createType = /CREATE\s+TYPE\s+(\S+)/i;
  const createTrigger = /CREATE\s+TRIGGER\s+(\S+)/i;
  const policy = /CREATE\s+POLICY\s+(\S+)/i;

  for (const ln of lines) {
    let m;

    if ((m = ln.match(addColumn))) {
      const trimmed = ln.trim();
      rollback.push('-- revert: ' + trimmed);
      rollback.push('ALTER TABLE ' + m[1] + ' DROP COLUMN IF EXISTS ' + m[2] + ';');
      continue;
    }
    if ((m = ln.match(createTable))) {
      const table = m[2]?.replace(/[";]/g,'');
      if (table) {
        const trimmed = ln.trim();
        rollback.push('-- revert: ' + trimmed);
        rollback.push('DROP TABLE IF EXISTS ' + table + ' CASCADE;');
      }
      continue;
    }
    if ((m = ln.match(createIndex))) {
      const idx = m[2]?.replace(/[";]/g,'');
      if (idx) {
        const trimmed = ln.trim();
        rollback.push('-- revert: ' + trimmed);
        rollback.push('DROP INDEX IF EXISTS ' + idx + ' CASCADE;');
      }
      continue;
    }
    if ((m = ln.match(createFn))) {
      const name = m[1]?.replace(/[";(].*$/,'');
      if (name) {
        const trimmed = ln.trim();
        rollback.push('-- revert: ' + trimmed);
        rollback.push('DROP FUNCTION IF EXISTS ' + name + ' CASCADE;');
      }
      continue;
    }
    if ((m = ln.match(createType))) {
      const name = m[1]?.replace(/[";(].*$/,'');
      if (name) {
        const trimmed = ln.trim();
        rollback.push('-- revert: ' + trimmed);
        rollback.push('DROP TYPE IF EXISTS ' + name + ' CASCADE;');
      }
      continue;
    }
    if ((m = ln.match(createTrigger))) {
      const name = m[1]?.replace(/[";(].*$/,'');
      if (name) {
        const trimmed = ln.trim();
        rollback.push('-- revert: ' + trimmed);
        rollback.push('DROP TRIGGER IF EXISTS ' + name + ' ON public; -- adjust schema if needed');
      }
      continue;
    }
    if ((m = ln.match(policy))) {
      const name = m[1]?.replace(/[";(].*$/,'');
      if (name) {
        const trimmed = ln.trim();
        rollback.push('-- revert: ' + trimmed);
        rollback.push('DROP POLICY IF EXISTS ' + name + ' ON public.*; -- set table');
      }
      continue;
    }
  }

  return rollback;
}

function main() {
  const files = readdirSync(MIGR_DIR).filter(f => f.endsWith('.sql')).sort().reverse();
  const out: string[] = [
    '-- Auto-generated rollback (heuristic)',
    '-- Generated: ' + new Date().toISOString(),
    '',
    'BEGIN;',
  ];

  for (const f of files) {
    const sql = readFileSync(join(MIGR_DIR, f), 'utf-8');
    out.push('\n-- from: ' + f);
    out.push(...extractRollback(sql));
  }

  out.push('COMMIT;', '');
  writeFileSync(OUT, out.join('\n'));
  console.log('✅ Rollback script generated → ' + OUT);
}

try {
  mkdirSync(MIGR_DIR, { recursive: true });
  main();
  process.exit(0);
} catch (e: any) {
  console.error('❌ migrate-rollback failed:', e.message);
  process.exit(1);
}

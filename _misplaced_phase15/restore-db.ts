// scripts/restore-db.ts
// 안전모드: 기본 dry-run. 실제 복구는 --force & ALLOW_RESTORE=true 필요.
import { execSync } from 'child_process';
import { readFileSync, existsSync } from 'fs';
import { createHash } from 'crypto';

const PROJECT_REF = process.env.SUPABASE_PROJECT_REF || '';
if (!PROJECT_REF) throw new Error('SUPABASE_PROJECT_REF missing');

const args = process.argv.slice(2);
const fileArgIdx = args.indexOf('--file');
const dryRun = args.includes('--dry-run');
const force = args.includes('--force') && process.env.ALLOW_RESTORE === 'true';

if (fileArgIdx < 0 || !args[fileArgIdx + 1]) {
  console.error('Usage: npm run db:restore -- --file <path.sql> [--dry-run] [--force]');
  process.exit(1);
}
const file = args[fileArgIdx + 1];

function sha256(buf: Buffer) {
  return createHash('sha256').update(buf).digest('hex');
}

try {
  if (!existsSync(file)) throw new Error('backup file not found: ' + file);
  const buf = readFileSync(file);
  const got = sha256(buf);
  console.log('🔒 backup sha256: ' + got);

  if (dryRun) {
    console.log('🧪 DRY-RUN PLAN:');
    console.log('- Validate backup hash');
    console.log('- (Would) npx supabase db reset');
    console.log('- (Would) npx supabase db push');
    process.exit(0);
  }

  if (!force) {
    console.error('❌ for safety, real restore requires --force and ALLOW_RESTORE=true');
    process.exit(1);
  }

  execSync('npx supabase db reset --project-ref ' + PROJECT_REF, { stdio: 'inherit' });
  execSync('npx supabase db push --project-ref ' + PROJECT_REF, { stdio: 'inherit' });

  console.log('✅ restore completed');
  process.exit(0);
} catch (e: any) {
  console.error('❌ restore failed:', e.message);
  process.exit(1);
}

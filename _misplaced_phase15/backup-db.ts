// scripts/backup-db.ts
import { execSync } from 'child_process';
import { writeFileSync, mkdirSync, readFileSync } from 'fs';
import { join } from 'path';
import { createHash } from 'crypto';
import { gzipSync } from 'zlib';

const PROJECT_REF = process.env.SUPABASE_PROJECT_REF || '';
if (!PROJECT_REF) throw new Error('SUPABASE_PROJECT_REF missing');

const BACKUP_DIR = join(process.cwd(), 'backups');
mkdirSync(BACKUP_DIR, { recursive: true });

function sha256(buf: Buffer) {
  return createHash('sha256').update(buf).digest('hex');
}

function run(cmd: string) {
  return execSync(cmd, { encoding: 'utf-8', stdio: 'pipe' });
}

const ts = new Date().toISOString().replace(/[:.]/g,'-');
const rawFile = join(BACKUP_DIR, 'backup-' + ts + '.sql');
const wantGzip = process.argv.includes('--compress');

try {
  // remote schema-only dump (안전)
  run('npx supabase db dump --project-ref ' + PROJECT_REF + ' --data-only=false --db-url "" --schema public --file ' + rawFile);
  const buf = readFileSync(rawFile);
  const hash = sha256(buf);
  const shaFile = rawFile + '.sha256';
  writeFileSync(shaFile, hash);

  if (wantGzip) {
    const gz = gzipSync(buf);
    const gzFile = rawFile + '.gz';
    writeFileSync(gzFile, gz);
  }

  writeFileSync(rawFile + '.meta.json', JSON.stringify({
    projectRef: PROJECT_REF,
    createdAt: new Date().toISOString(),
    bytes: buf.length,
    sha256: hash,
    compress: wantGzip
  }, null, 2));

  console.log('✅ Backup created: ' + rawFile);
  console.log('🔒 SHA-256: ' + hash);
  process.exit(0);
} catch (e: any) {
  console.error('❌ backup failed:', e.message);
  process.exit(1);
}

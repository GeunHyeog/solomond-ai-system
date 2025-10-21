// scripts/migrate-verify.ts
import { createHash } from 'crypto';
import { readFileSync, existsSync } from 'fs';
import { join } from 'path';

const MIGRATIONS_DIR = join(process.cwd(), 'supabase', 'migrations');
const REGISTRY_FILE = join(MIGRATIONS_DIR, '.registry.json');

type Entry = { file: string; sha256: string };

function sha256(buf: Buffer) {
  return createHash('sha256').update(buf).digest('hex');
}

function main() {
  if (!existsSync(REGISTRY_FILE)) {
    console.error('❌ Registry not found. Run: npm run migrate:hash');
    process.exit(1);
  }
  const registry: Entry[] = JSON.parse(readFileSync(REGISTRY_FILE, 'utf-8'));

  let errors = 0;
  for (const row of registry) {
    const p = join(MIGRATIONS_DIR, row.file);
    if (!existsSync(p)) {
      console.error(`❌ Missing file: ${row.file}`);
      errors++;
      continue;
    }
    const got = sha256(readFileSync(p));
    if (got !== row.sha256) {
      console.error(`❌ Checksum mismatch: ${row.file}\n   expected: ${row.sha256}\n   got     : ${got}`);
      errors++;
    } else {
      console.log(`✅ OK: ${row.file}`);
    }
  }

  if (errors > 0) {
    console.error(`\n❌ Verification failed: ${errors}`);
    process.exit(1);
  }
  console.log(`\n✅ All ${registry.length} migrations verified`);
}

try {
  main();
  process.exit(0);
} catch (e: any) {
  console.error('❌ migrate-verify failed:', e.message);
  process.exit(1);
}

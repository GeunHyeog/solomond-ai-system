// scripts/migrate-hash.ts
import { createHash } from 'crypto';
import { readdirSync, readFileSync, statSync, writeFileSync, mkdirSync, existsSync } from 'fs';
import { join, extname } from 'path';

const MIGRATIONS_DIR = join(process.cwd(), 'supabase', 'migrations');
const REGISTRY_FILE = join(MIGRATIONS_DIR, '.registry.json');

type Entry = { file: string; sha256: string; bytes: number; mtime: string };

function sha256(buf: Buffer) {
  return createHash('sha256').update(buf).digest('hex');
}

function main() {
  if (!existsSync(MIGRATIONS_DIR)) {
    throw new Error(`Migrations dir not found: ${MIGRATIONS_DIR}`);
  }
  const files = readdirSync(MIGRATIONS_DIR)
    .filter(f => extname(f) === '.sql')
    .sort();

  const registry: Entry[] = files.map(file => {
    const p = join(MIGRATIONS_DIR, file);
    const buf = readFileSync(p);
    const stats = statSync(p);
    return {
      file,
      sha256: sha256(buf),
      bytes: stats.size,
      mtime: stats.mtime.toISOString(),
    };
  });

  writeFileSync(REGISTRY_FILE, JSON.stringify(registry, null, 2));
  console.log(`✅ Registry updated: ${registry.length} file(s)\n→ ${REGISTRY_FILE}`);
}

try {
  mkdirSync(join(process.cwd(), 'supabase', 'migrations'), { recursive: true });
  main();
  process.exit(0);
} catch (e: any) {
  console.error('❌ migrate-hash failed:', e.message);
  process.exit(1);
}

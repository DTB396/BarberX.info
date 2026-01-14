#!/usr/bin/env node
import { execSync } from 'child_process';
import fs from 'fs';
import path from 'path';

function main() {
  const repoRoot = process.cwd();
  const since = process.argv[2] || '1.week';
  const log = execSync(`git log --since=${since} --pretty=format:%ad%x09%h%x09%s --date=short`, { encoding: 'utf-8' });
  const lines = log.split('\n').filter(Boolean);
  const outDir = path.join(repoRoot, 'reports');
  if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });
  const outPath = path.join(outDir, `changelog-${new Date().toISOString().slice(0,10)}.md`);
  const header = `# Weekly Public Record Update\n\nGenerated: ${new Date().toISOString()}\n\n`;
  const body = lines.map(l => {
    const [date, hash, subject] = l.split('\t');
    return `- ${date} · ${hash} · ${subject}`;
  }).join('\n');
  fs.writeFileSync(outPath, header + body + '\n', 'utf-8');
  console.log(`Wrote ${outPath}`);
}

main();

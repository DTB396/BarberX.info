#!/usr/bin/env node
import fs from 'fs';
import path from 'path';
import crypto from 'crypto';
import yaml from 'js-yaml';

function sha256File(filePath) {
  const hash = crypto.createHash('sha256');
  const stream = fs.createReadStream(filePath);
  return new Promise((resolve, reject) => {
    stream.on('data', chunk => hash.update(chunk));
    stream.on('error', reject);
    stream.on('end', () => resolve(hash.digest('hex')));
  });
}

function parseFrontMatter(md) {
  const match = md.match(/^---([\s\S]*?)---/);
  if (!match) return {};
  try {
    return yaml.load(match[1], { schema: yaml.JSON_SCHEMA }) || {};
  } catch {
    return {};
  }
}

async function main() {
  const repoRoot = process.cwd();
  const dataRoot = path.join(repoRoot, '_data', 'checksums');
  if (!fs.existsSync(dataRoot)) fs.mkdirSync(dataRoot, { recursive: true });

  const casesDir = path.join(repoRoot, '_cases');
  const slugs = fs.readdirSync(casesDir).filter(d => fs.statSync(path.join(casesDir, d)).isDirectory());

  for (const slug of slugs) {
    const indexMdPath = path.join(casesDir, slug, 'index.md');
    if (!fs.existsSync(indexMdPath)) continue;
    const md = fs.readFileSync(indexMdPath, 'utf-8');
    const fm = parseFrontMatter(md);
    const assetsDir = (fm.assets_dir || '').replace(/^\//, '');
    const docs = Array.isArray(fm.documents) ? fm.documents : [];

    if (!assetsDir) {
      // Try default normalized filings directory
      const guess = path.join('cases', slug, 'filings');
      if (fs.existsSync(path.join(repoRoot, guess))) {
        fm.assets_dir = `/${guess}/`;
      } else {
        continue;
      }
    }

    const absAssetsDir = path.join(repoRoot, fm.assets_dir.replace(/^\//, ''));
    if (!fs.existsSync(absAssetsDir)) continue;

    // If no docs declared, hash all files in assets dir
    let relFiles = [];
    if (docs.length) {
      relFiles = docs.map(d => d.path).filter(Boolean);
    } else {
      relFiles = fs.readdirSync(absAssetsDir).filter(f => fs.statSync(path.join(absAssetsDir, f)).isFile());
    }

    const checksums = {};
    for (const rel of relFiles) {
      const abs = path.join(absAssetsDir, rel);
      if (!fs.existsSync(abs)) continue;
      checksums[rel] = await sha256File(abs);
    }

    const outPath = path.join(dataRoot, `${slug}.yml`);
    const outYaml = yaml.dump(checksums, { lineWidth: -1 });
    fs.writeFileSync(outPath, outYaml, 'utf-8');
    console.log(`Wrote ${outPath}`);
  }
}

main();

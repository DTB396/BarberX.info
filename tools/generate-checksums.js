// Checksum and Provenance Tracking for FaithFrontier
// Auto-generates SHA256 checksums for all filings
// Stores checksums in _data/checksums/<slug>.yml

import fs from 'fs';
import path from 'path';
import crypto from 'crypto';
import yaml from 'js-yaml';

const CASES_DIR = 'cases';
const CHECKSUM_DIR = '_data/checksums';
const ensureDir = p => fs.existsSync(p) || fs.mkdirSync(p, { recursive: true });

function getSlugFromPath(filePath) {
  const parts = filePath.split(path.sep);
  const idx = parts.indexOf('cases');
  if (idx >= 0 && parts.length > idx + 2) return parts[idx + 1];
  return 'unknown';
}

function sha256File(filePath) {
  const hash = crypto.createHash('sha256');
  const data = fs.readFileSync(filePath);
  hash.update(data);
  return hash.digest('hex');
}

function main() {
  ensureDir(CHECKSUM_DIR);
  const bySlug = {};
  function walk(dir) {
    if (!fs.existsSync(dir)) return;
    for (const entry of fs.readdirSync(dir)) {
      const full = path.join(dir, entry);
      const stat = fs.statSync(full);
      if (stat.isDirectory()) walk(full);
      else if (entry.toLowerCase().endsWith('.pdf')) {
        const slug = getSlugFromPath(full);
        if (!bySlug[slug]) bySlug[slug] = {};
        bySlug[slug][path.basename(full)] = sha256File(full);
      }
    }
  }
  walk(CASES_DIR);
  for (const slug in bySlug) {
    const outPath = path.join(CHECKSUM_DIR, `${slug}.yml`);
    fs.writeFileSync(outPath, yaml.dump(bySlug[slug]));
    console.log(`Checksums written: ${outPath}`);
  }
  console.log('Checksum generation complete.');
}

main();

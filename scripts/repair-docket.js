// repair-docket.js
// Audits and repairs case folders: ensures all PDFs in cases/<slug>/filings/ are registered in _data/docket/<slug>.yml
// Usage: node scripts/repair-docket.js

import fs from 'fs';
import path from 'path';
import yaml from 'js-yaml';

const CASES_DIR = 'cases';
const DOCKET_DIR = '_data/docket';

const ensureDir = p => fs.mkdirSync(p, { recursive: true });
const readYml = p => fs.existsSync(p) ? yaml.load(fs.readFileSync(p, 'utf8')) : [];
const writeYml = (p, obj) => fs.writeFileSync(p, yaml.dump(obj, { lineWidth: 1000 }));

const guessType = name => {
  const n = name.toLowerCase();
  if (n.includes('order')) return 'Order';
  if (n.includes('notice')) return 'Notice';
  if (n.includes('brief')) return 'Brief';
  if (n.includes('exhibit')) return 'Exhibit';
  if (n.includes('motion')) return 'Motion';
  return 'Filing';
};

const guessDate = (name, statsMtime) => {
  const m = name.match(/(20\d{2})[-_\.]?(0[1-9]|1[0-2])[-_\.]?(0[1-9]|[12]\d|3[01])/);
  if (m) return `${m[1]}-${m[2]}-${m[3]}`;
  const d = new Date(statsMtime);
  return d.toISOString().slice(0,10);
};

const repairDocket = slug => {
  const docketDir = path.join(CASES_DIR, slug, 'filings');
  if (!fs.existsSync(docketDir)) return;
  const files = fs.readdirSync(docketDir).filter(f => f.toLowerCase().endsWith('.pdf'));
  const docketFile = path.join(DOCKET_DIR, `${slug}.yml`);
  let entries = readYml(docketFile) || [];
  let changed = false;

  for (const file of files) {
    const filePath = path.join(docketDir, file);
    const stats = fs.statSync(filePath);
    const id = file.replace(/\.pdf$/i, '').slice(0,64);
    const date = guessDate(file, stats.mtimeMs);
    const type = guessType(file);
    const title = file.replace(/\.pdf$/i,'').replace(/[_-]+/g,' ').replace(/\s+/g,' ').trim();
    const fileUrl = `/cases/${slug}/filings/${file}`;
    if (!entries.find(e => e.file === fileUrl || e.file === file)) {
      entries.push({ id, date, type, title, file: fileUrl, notes: 'Repaired: auto-registered' });
      changed = true;
    }
  }
  if (changed) {
    entries = entries.sort((a,b)=> (a.date < b.date ? 1 : -1));
    ensureDir(DOCKET_DIR);
    writeYml(docketFile, entries);
    console.log(`Updated docket for ${slug}: ${entries.length} entries.`);
  }
};

const main = () => {
  const slugs = fs.readdirSync(CASES_DIR).filter(f => fs.existsSync(path.join(CASES_DIR, f, 'filings')));
  for (const slug of slugs) {
    repairDocket(slug);
  }
  console.log('Repair complete.');
};

main();

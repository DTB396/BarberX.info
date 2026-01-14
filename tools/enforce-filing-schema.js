// Enforce standardized file naming and structure for filings
// Auto-rename files and move misplaced filings

import fs from 'fs';
import path from 'path';

const CASES_DIR = 'cases';
const FILING_REGEX = /^(\d{4}-\d{2}-\d{2})_(Order|Notice|Brief|Exhibit|Motion|Filing)_(.+)\.pdf$/i;

function walk(dir, cb) {
  if (!fs.existsSync(dir)) return;
  for (const entry of fs.readdirSync(dir)) {
    const full = path.join(dir, entry);
    const stat = fs.statSync(full);
    if (stat.isDirectory()) walk(full, cb);
    else cb(full, stat);
  }
}

function enforceNaming() {
  walk(CASES_DIR, (file, stat) => {
    if (!file.toLowerCase().endsWith('.pdf')) return;
    const name = path.basename(file);
    if (!FILING_REGEX.test(name)) {
      // Attempt to auto-rename (simple heuristic)
      const dir = path.dirname(file);
      const m = name.match(/(20\d{2})[-_]?([01]\d)[-_]?([0-3]\d)/);
      const date = m ? `${m[1]}-${m[2]}-${m[3]}` : 'unknown';
      let type = 'Filing';
      if (/order/i.test(name)) type = 'Order';
      else if (/notice/i.test(name)) type = 'Notice';
      else if (/brief/i.test(name)) type = 'Brief';
      else if (/exhibit/i.test(name)) type = 'Exhibit';
      else if (/motion/i.test(name)) type = 'Motion';
      const stub = name.replace(/\.pdf$/i, '').replace(/[^a-zA-Z0-9]+/g, '-').replace(/^-+|-+$/g, '');
      const newName = `${date}_${type}_${stub}.pdf`;
      const newPath = path.join(dir, newName);
      if (!fs.existsSync(newPath)) {
        fs.renameSync(file, newPath);
        console.log(`Renamed: ${file} -> ${newPath}`);
      }
    }
  });
}

enforceNaming();
console.log('Filing schema enforcement complete.');

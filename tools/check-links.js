// Automated Link Checking for FaithFrontier
// Scans case pages and dockets for broken file links
// Reports missing or invalid references

import fs from 'fs';
import path from 'path';
import yaml from 'js-yaml';

const CASES_INDEX = '_cases_index';
const DOCKET_INDEX = '_data/docket_index';
const CASES_DIR = 'cases';
const REPORT_DIR = 'reports';

const ensureDir = p => fs.existsSync(p) || fs.mkdirSync(p, { recursive: true });

function checkDocketLinks() {
  const issues = [];
  for (const entry of fs.readdirSync(DOCKET_INDEX)) {
    if (!entry.endsWith('.yml')) continue;
    const filePath = path.join(DOCKET_INDEX, entry);
    const data = yaml.load(fs.readFileSync(filePath, 'utf8'));
    for (const item of data) {
      if (!item.file) continue;
      const repoPath = item.file.startsWith('/') ? item.file.slice(1) : item.file;
      if (!fs.existsSync(repoPath)) {
        issues.push({ docket: filePath, missing: repoPath });
      }
    }
  }
  return issues;
}

function main() {
  ensureDir(REPORT_DIR);
  const issues = checkDocketLinks();
  fs.writeFileSync(path.join(REPORT_DIR, 'link-check-report.json'), JSON.stringify(issues, null, 2));
  console.log('Link check complete. See reports/link-check-report.json');
}

main();

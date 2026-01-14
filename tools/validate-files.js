// Automated File Validation for FaithFrontier
// Scans for missing, duplicate, or zero-byte PDFs
// Validates YAML front matter in case/docket files
// Generates a report for quick fixes

import fs from 'fs';
import path from 'path';
import yaml from 'js-yaml';

const CASES_INDEX = '_cases_index';
const DOCKET_INDEX = '_data/docket_index';
const CASES_DIR = 'cases';
const REPORT_DIR = 'reports';
const PDF_MIN_BYTES = 100; // Consider <100 bytes as placeholder/empty

const ensureDir = p => fs.existsSync(p) || fs.mkdirSync(p, { recursive: true });

function findAllPdfs() {
  const pdfs = [];
  function walk(dir) {
    if (!fs.existsSync(dir)) return;
    for (const entry of fs.readdirSync(dir)) {
      const full = path.join(dir, entry);
      const stat = fs.statSync(full);
      if (stat.isDirectory()) walk(full);
      else if (entry.toLowerCase().endsWith('.pdf')) pdfs.push({ path: full, size: stat.size });
    }
  }
  walk(CASES_DIR);
  return pdfs;
}

function findDuplicates(files) {
  const seen = new Map();
  const dups = [];
  for (const f of files) {
    const name = path.basename(f.path).toLowerCase();
    if (seen.has(name)) dups.push([seen.get(name), f.path]);
    else seen.set(name, f.path);
  }
  return dups;
}

function findZeroByte(files) {
  return files.filter(f => f.size < PDF_MIN_BYTES);
}

function validateYamlFrontMatter(filePath) {
  try {
    const content = fs.readFileSync(filePath, 'utf8');
    const match = content.match(/^---\n([\s\S]*?)\n---/);
    if (!match) return { valid: false, error: 'Missing YAML front matter' };
    const data = yaml.load(match[1]);
    // Check required fields
    const required = ['title', 'status', 'primary_docket'];
    for (const key of required) {
      if (!data[key]) return { valid: false, error: `Missing required field: ${key}` };
    }
    return { valid: true };
  } catch (e) {
    return { valid: false, error: e.message };
  }
}

function scanCaseFiles() {
  const results = [];
  for (const entry of fs.readdirSync(CASES_INDEX)) {
    if (!entry.endsWith('.md')) continue;
    const filePath = path.join(CASES_INDEX, entry);
    const res = validateYamlFrontMatter(filePath);
    results.push({ file: filePath, ...res });
  }
  return results;
}

function scanDocketFiles() {
  const results = [];
  for (const entry of fs.readdirSync(DOCKET_INDEX)) {
    if (!entry.endsWith('.yml')) continue;
    const filePath = path.join(DOCKET_INDEX, entry);
    try {
      yaml.load(fs.readFileSync(filePath, 'utf8'));
      results.push({ file: filePath, valid: true });
    } catch (e) {
      results.push({ file: filePath, valid: false, error: e.message });
    }
  }
  return results;
}

function main() {
  ensureDir(REPORT_DIR);
  const pdfs = findAllPdfs();
  const duplicates = findDuplicates(pdfs);
  const zeroByte = findZeroByte(pdfs);
  const caseYaml = scanCaseFiles();
  const docketYaml = scanDocketFiles();

  const report = {
    timestamp: new Date().toISOString(),
    pdfs: pdfs.length,
    duplicates,
    zeroByte,
    caseYaml,
    docketYaml
  };
  fs.writeFileSync(path.join(REPORT_DIR, 'file-validation-report.json'), JSON.stringify(report, null, 2));
  console.log('Validation complete. See reports/file-validation-report.json');
}

main();

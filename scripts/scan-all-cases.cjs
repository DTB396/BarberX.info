// FaithFrontier Universal Case Folder Scanner & Compliance Agent
// Scans all folders under /cases for PDFs, normalizes, analyzes, and updates dockets/README
// Requires: npm install pdf-parse js-yaml

const fs = require('fs');
const path = require('path');
const pdf = require('pdf-parse');
const yaml = require('js-yaml');

const CASES_ROOT = path.join(__dirname, '../cases');

function getAllCaseFolders(root) {
  return fs.readdirSync(root).filter(f => fs.statSync(path.join(root, f)).isDirectory());
}

function getAllFilesRecursive(dir, ext = '.pdf') {
  let results = [];
  fs.readdirSync(dir).forEach(f => {
    const full = path.join(dir, f);
    if (fs.statSync(full).isDirectory()) {
      results = results.concat(getAllFilesRecursive(full, ext));
    } else if (f.toLowerCase().endsWith(ext)) {
      results.push(full);
    }
  });
  return results;
}

function normalizeFilename(filename) {
  // Try to extract date and doc type from filename, else fallback to original
  const re = /(\d{4})[-_]?(\d{2})[-_]?(\d{2})[-_]?([\w-]+)?/;
  const match = filename.match(re);
  if (match) {
    const date = `${match[1]}${match[2]}${match[3]}`;
    const rest = filename.replace(re, '').replace(/^[^a-zA-Z0-9]+/, '').replace(/\.pdf$/i, '');
    return `${match[1]}-${match[2]}-${match[3]}${rest ? '-' + rest : ''}.pdf`;
  }
  return filename;
}

function updateDocketYaml(caseDir, filings) {
  const ymlPath = path.join(caseDir, 'docket.yml');
  let entries = [];
  if (fs.existsSync(ymlPath)) {
    entries = yaml.load(fs.readFileSync(ymlPath, 'utf8')) || [];
  }
  // Remove old entries for files no longer present
  entries = entries.filter(e => filings.some(f => f.file === e.file));
  // Add new filings
  for (const f of filings) {
    if (!entries.some(e => e.file === f.file)) {
      entries.push({
        id: `${f.file.replace(/\.pdf$/, '')}`,
        date: f.date || '',
        type: f.type || '',
        title: f.title || '',
        file: f.file
      });
    }
  }
  fs.writeFileSync(ymlPath, yaml.dump(entries, { lineWidth: 120 }));
}

function updateReadme(caseDir, filings) {
  const readmePath = path.join(caseDir, 'README.md');
  let md = `# ${path.basename(caseDir)}\n\n## Filings Index\n`;
  for (const f of filings) {
    md += `- [${f.file}](./${path.relative(caseDir, f.fullPath)})\n`;
  }
  fs.writeFileSync(readmePath, md);
}

async function analyzeAndNormalizeCase(caseDir) {
  const pdfs = getAllFilesRecursive(caseDir);
  let filings = [];
  for (const pdfPath of pdfs) {
    const origName = path.basename(pdfPath);
    const normName = normalizeFilename(origName);
    const normPath = path.join(path.dirname(pdfPath), normName);
    if (origName !== normName) {
      fs.renameSync(pdfPath, normPath);
    }
    let meta = { file: normName, fullPath: normPath };
    try {
      const data = await pdf(fs.readFileSync(normPath));
      meta.title = data.info && data.info.Title ? data.info.Title : '';
      meta.date = (normName.match(/\d{4}-\d{2}-\d{2}/) || [])[0] || '';
      meta.type = meta.title || '';
    } catch (e) {
      meta.title = '';
    }
    filings.push(meta);
  }
  updateDocketYaml(caseDir, filings);
  updateReadme(caseDir, filings);
  console.log(`Updated: ${caseDir}`);
}

async function main() {
  const caseFolders = getAllCaseFolders(CASES_ROOT);
  for (const folder of caseFolders) {
    await analyzeAndNormalizeCase(path.join(CASES_ROOT, folder));
  }
  console.log('All case folders scanned and updated.');
}

main();

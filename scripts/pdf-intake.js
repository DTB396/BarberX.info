// FaithFrontier PDF Intake & Docket Automation (legacy)
// Note: The canonical intake is scripts/docket-intake.js. This script is kept for compatibility.

import fs from 'node:fs';
import path from 'node:path';
import process from 'node:process';
import pdf from 'pdf-parse';
import yaml from 'js-yaml';

const repoRoot = process.cwd();
const INBOX = path.join(repoRoot, '_inbox');
const CASES_DIR = path.join(repoRoot, 'cases');
const DOCKET_DATA_DIR = path.join(repoRoot, '_data', 'docket');

// Helper: Extract metadata from filename
function parseFilename(filename) {
  // Example: 2025-11-12_Filing_Written-Appearance-ADA-MTD.pdf
  const re = /^(\d{4}-\d{2}-\d{2})_([\w-]+)_(.+)\.pdf$/i;
  const match = filename.match(re);
  if (match) {
    return {
      date: match[1],
      type: match[2].replace(/-/g, ' '),
      short: match[3].replace(/-/g, ' '),
      complete: true
    };
  }
  return { complete: false };
}

// Helper: Extract metadata from PDF text
function extractMetaFromText(text) {
  // Simple heuristics: look for date, title/type in first 20 lines
  const lines = text.split(/\r?\n/).slice(0, 20);
  let date = null, type = null, short = null;
  for (const line of lines) {
    if (!date) {
      const d = line.match(/(\d{4}-\d{2}-\d{2})/);
      if (d) date = d[1];
    }
    if (!type && /order|motion|complaint|notice|filing|judgment|appearance|petition|application/i.test(line)) {
      type = line.trim().split(' ')[0];
    }
    if (!short && line.length > 10 && line.length < 80) {
      short = line.trim();
    }
  }
  return { date, type, short };
}

// Helper: Build new filename
function buildFilename(meta) {
  let date = meta.date || 'unknown';
  let type = meta.type ? meta.type.replace(/\s+/g, '-') : 'Document';
  let short = meta.short ? meta.short.replace(/\s+/g, '-') : 'Untitled';
  return `${date}_${type}_${short}.pdf`;
}

// Helper: Update YAML docket file
function updateDocketYaml(caseSlug, meta, filename) {
  const ymlPath = path.join(DOCKET_DATA_DIR, `${caseSlug}.yml`);
  let entries = [];
  if (fs.existsSync(ymlPath)) {
    entries = yaml.load(fs.readFileSync(ymlPath, 'utf8')) || [];
  }
  // Generate unique id
  const id = `${meta.date || 'unknown'}-${meta.type || 'doc'}-${Math.random().toString(36).slice(2, 8)}`;
  entries.push({
    id,
    date: meta.date || '',
    type: meta.type || '',
    title: meta.short || '',
    file: `/cases/${caseSlug}/filings/${filename}`
  });
  fs.writeFileSync(ymlPath, yaml.dump(entries, { lineWidth: 120 }));
}

// Helper: Update or create Markdown summary (optional)
function updateOrCreateMarkdown(caseSlug, meta, filename) {
  const caseDir = path.join(CASES_DIR, caseSlug);
  if (!fs.existsSync(caseDir)) fs.mkdirSync(caseDir, { recursive: true });
  const mdPath = path.join(caseDir, 'docket.md');
  let md = '';
  if (fs.existsSync(mdPath)) {
    md = fs.readFileSync(mdPath, 'utf8');
  }
  md += `\n- **${meta.date || 'Unknown'}**: [${meta.short || filename}](/cases/${caseSlug}/filings/${filename})\n`;
  fs.writeFileSync(mdPath, md);
}

// Main intake process
async function processInbox() {
  const files = fs.readdirSync(INBOX).filter(f => f.toLowerCase().endsWith('.pdf'));
  for (const file of files) {
    const filePath = path.join(INBOX, file);
    let meta = parseFilename(file);
    let text = '';
    if (!meta.complete) {
      try {
        const data = await pdf(fs.readFileSync(filePath));
        text = data.text;
        meta = { ...meta, ...extractMetaFromText(text) };
      } catch (e) {
        console.error(`Failed to parse PDF: ${file}`, e);
      }
    }
    // Ask user for case slug or infer from folder/filename (customize as needed)
    let caseSlug = 'general'; // TODO: improve slug inference or prompt user
    // Organize PDFs by case
    const caseDocketDir = path.join(CASES_DIR, caseSlug, 'filings');
    if (!fs.existsSync(caseDocketDir)) fs.mkdirSync(caseDocketDir, { recursive: true });
    const newName = buildFilename(meta);
    const destPath = path.join(caseDocketDir, newName);
    fs.renameSync(filePath, destPath);
    updateDocketYaml(caseSlug, meta, newName);
    updateOrCreateMarkdown(caseSlug, meta, newName);
    console.log(`Processed: ${file} -> ${destPath}`);
  }
}

processInbox().then(() => console.log('Intake complete.')).catch(console.error);

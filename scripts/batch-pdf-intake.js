// FaithFrontier Batch PDF Upload & Case Initiative Dashboard (legacy)
// Note: The canonical intake is scripts/docket-intake.js. This script is kept for compatibility.

import fs from 'node:fs';
import path from 'node:path';
import process from 'node:process';
import { createInterface } from 'node:readline/promises';
import { stdin as input, stdout as output } from 'node:process';
import pdf from 'pdf-parse';
import yaml from 'js-yaml';

const repoRoot = process.cwd();

const CASES_MAP_PATH = path.join(repoRoot, '_data', 'cases-map.yml');
const LOG_PATH = path.join(repoRoot, 'reports', 'intake-log.txt');
const REVIEW_PATH = path.join(repoRoot, 'reports', 'intake-review.md');

const INBOX = path.join(repoRoot, '_inbox');
const CASES_DIR = path.join(repoRoot, 'cases');
const DOCKET_DATA_DIR = path.join(repoRoot, '_data', 'docket');
const DASHBOARD_MD = path.join(repoRoot, '_site', 'case-initiatives.md');

// Helper: Extract metadata from filename
function parseFilename(filename) {
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

function buildFilename(meta) {
  let date = meta.date || 'unknown';
  let type = meta.type ? meta.type.replace(/\s+/g, '-') : 'Document';
  let short = meta.short ? meta.short.replace(/\s+/g, '-') : 'Untitled';
  return `${date}_${type}_${short}.pdf`;
}

function updateDocketYaml(caseSlug, meta, filename) {
  const ymlPath = path.join(DOCKET_DATA_DIR, `${caseSlug}.yml`);
  let entries = [];
  if (fs.existsSync(ymlPath)) {
    entries = yaml.load(fs.readFileSync(ymlPath, 'utf8')) || [];
  }
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

function updateDashboard(caseSlugs) {
  let dashboard = `# Current Case Initiatives\n\n`;
  for (const slug of caseSlugs) {
    const ymlPath = path.join(DOCKET_DATA_DIR, `${slug}.yml`);
    if (fs.existsSync(ymlPath)) {
      const entries = yaml.load(fs.readFileSync(ymlPath, 'utf8')) || [];
      dashboard += `## ${slug.replace(/-/g, ' ').toUpperCase()}\n`;
      for (const entry of entries) {
        const fileUrl = entry.file?.startsWith('/') ? entry.file : `/cases/${slug}/filings/${entry.file}`;
        dashboard += `- **${entry.date}**: ${entry.title} ([PDF](${fileUrl}))\n`;
      }
      dashboard += '\n';
    }
  }
  fs.writeFileSync(DASHBOARD_MD, dashboard);
}


// Load known case slugs from cases-map.yml
function loadCaseSlugs() {
  if (fs.existsSync(CASES_MAP_PATH)) {
    const map = yaml.load(fs.readFileSync(CASES_MAP_PATH, 'utf8'));
    return Object.keys(map || {});
  }
  return [];
}

// Try to auto-detect case slug from filename or PDF text
function autoDetectCaseSlug(meta, text, knownSlugs) {
  const candidates = [];
  for (const slug of knownSlugs) {
    if (meta.short && slug.replace(/-/g, '').toLowerCase().includes(meta.short.replace(/\s+/g, '').toLowerCase())) {
      candidates.push(slug);
    }
    if (text && text.toLowerCase().includes(slug.replace(/-/g, '').toLowerCase())) {
      candidates.push(slug);
    }
  }
  return candidates.length === 1 ? candidates[0] : null;
}

// Check for duplicate document in case docket
function isDuplicate(caseSlug, meta, filename) {
  const ymlPath = path.join(DOCKET_DATA_DIR, `${caseSlug}.yml`);
  if (!fs.existsSync(ymlPath)) return false;
  const entries = yaml.load(fs.readFileSync(ymlPath, 'utf8')) || [];
  return entries.some(e => e.date === meta.date && e.type === meta.type && e.title === meta.short);
}

// Log actions and errors
function logAction(msg) {
  fs.appendFileSync(LOG_PATH, `[${new Date().toISOString()}] ${msg}\n`);
}

// Collect review items
let reviewItems = [];

// Improved metadata extraction (try PDF metadata fields)
function extractMetaFromText(text, pdfData) {
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
  // Try PDF metadata
  if (pdfData && pdfData.info) {
    if (!date && pdfData.info.CreationDate) {
      const m = pdfData.info.CreationDate.match(/(\d{4})(\d{2})(\d{2})/);
      if (m) date = `${m[1]}-${m[2]}-${m[3]}`;
    }
    if (!short && pdfData.info.Title) short = pdfData.info.Title;
  }
  return { date, type, short };
}


async function processInboxBatch() {
  const files = fs.readdirSync(INBOX).filter(f => f.toLowerCase().endsWith('.pdf'));
  let caseSlugs = new Set();
  const knownSlugs = loadCaseSlugs();
  const rl = createInterface({ input, output });
  for (const file of files) {
    const filePath = path.join(INBOX, file);
    let meta = parseFilename(file);
    let text = '', pdfData = null;
    if (!meta.complete) {
      try {
        pdfData = await pdf(fs.readFileSync(filePath));
        text = pdfData.text;
        meta = { ...meta, ...extractMetaFromText(text, pdfData) };
      } catch (e) {
        logAction(`Failed to parse PDF: ${file} - ${e}`);
      }
    }
    // Auto-detect case slug
    let caseSlug = autoDetectCaseSlug(meta, text, knownSlugs);
    if (!caseSlug) {
      const defaultSlug = meta.short ? meta.short.toLowerCase().replace(/\s+/g, '-') : 'general';
      const answer = await rl.question(`Enter case slug (e.g., atl-l-002794-25) [${defaultSlug}]: `);
      caseSlug = (answer || defaultSlug).trim();
    }
    caseSlugs.add(caseSlug);
    const caseDocketDir = path.join(CASES_DIR, caseSlug, 'filings');
    if (!fs.existsSync(caseDocketDir)) fs.mkdirSync(caseDocketDir, { recursive: true });
    let newName = buildFilename(meta);
    // Duplicate/versioning logic
    if (isDuplicate(caseSlug, meta, newName)) {
      let base = newName.replace(/\.pdf$/, '');
      let v = 2;
      while (fs.existsSync(path.join(caseDocketDir, `${base}-v${v}.pdf`))) v++;
      newName = `${base}-v${v}.pdf`;
      logAction(`Duplicate detected for ${file}, saved as ${newName}`);
    }
    const destPath = path.join(caseDocketDir, newName);
    fs.renameSync(filePath, destPath);
    updateDocketYaml(caseSlug, meta, newName);
    updateOrCreateMarkdown(caseSlug, meta, newName);
    logAction(`Processed: ${file} -> ${destPath}`);
    // Review dashboard for ambiguous/missing fields
    if (!meta.date || !meta.type || !meta.short) {
      reviewItems.push({ file, caseSlug, meta, note: 'Missing/ambiguous metadata' });
    }
  }
  rl.close();
  updateDashboard(Array.from(caseSlugs));
  // Write review dashboard
  let reviewMd = '# Intake Review Dashboard\n\n';
  for (const item of reviewItems) {
    reviewMd += `- **${item.file}** (case: ${item.caseSlug}) - ${item.note}\n`;
  }
  fs.writeFileSync(REVIEW_PATH, reviewMd);
  logAction('Review dashboard updated: ' + REVIEW_PATH);
  console.log('Dashboard updated: ' + DASHBOARD_MD);
  console.log('Review dashboard: ' + REVIEW_PATH);
}

processInboxBatch().then(() => console.log('Batch intake complete.')).catch(console.error);

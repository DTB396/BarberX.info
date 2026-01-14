#!/usr/bin/env node
// Create inbox directories for all existing cases
// Usage: node scripts/create-case-inboxes.js

import fs from 'fs';
import path from 'path';
import yaml from 'js-yaml';

const INBOX_ROOT = '_inbox';
const CASES_MAP_FILE = '_data/cases-map.yml';
const CASES_DIR = '_cases';

// Ensure main inbox exists
if (!fs.existsSync(INBOX_ROOT)) {
  fs.mkdirSync(INBOX_ROOT, { recursive: true });
  console.log(`✓ Created ${INBOX_ROOT}/`);
}

// Load cases map
let casesMap = {};
if (fs.existsSync(CASES_MAP_FILE)) {
  casesMap = yaml.load(fs.readFileSync(CASES_MAP_FILE, 'utf8')) || {};
}

// Get unique case slugs from map
const slugs = [...new Set(Object.values(casesMap))];

console.log('\n📁 Creating case-specific inbox directories...\n');

for (const slug of slugs) {
  const inboxDir = path.join(INBOX_ROOT, slug);
  
  if (fs.existsSync(inboxDir)) {
    console.log(`  ⏭️  ${slug}/ (already exists)`);
    continue;
  }
  
  fs.mkdirSync(inboxDir, { recursive: true });
  
  // Create a .gitkeep to track empty directories
  fs.writeFileSync(path.join(inboxDir, '.gitkeep'), '');
  
  console.log(`  ✓ ${slug}/`);
}

console.log(`\n✅ Case inbox structure ready!`);
console.log(`\nUsage:`);
console.log(`  cp ~/Downloads/*.pdf _inbox/<case-slug>/`);
console.log(`  git add _inbox/ && git commit -m "intake: batch upload" && git push`);
console.log(`\nOr drop PDFs directly in _inbox/ root - they'll be auto-routed.\n`);

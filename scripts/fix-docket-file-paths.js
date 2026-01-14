#!/usr/bin/env node
// Fix docket YAML file paths to use absolute /cases/<slug>/filings/ format
// Usage: node scripts/fix-docket-paths.js [case-slug]

import fs from 'fs';
import path from 'path';
import yaml from 'js-yaml';

const DOCKET_DIR = '_data/docket';
const CASES_DIR = 'cases';

const args = process.argv.slice(2);
const targetSlug = args[0];

// Get all docket files
let docketFiles = [];
if (targetSlug) {
  const file = path.join(DOCKET_DIR, `${targetSlug}.yml`);
  if (fs.existsSync(file)) {
    docketFiles.push(file);
  } else {
    console.error(`❌ Docket file not found: ${file}`);
    process.exit(1);
  }
} else {
  docketFiles = fs.readdirSync(DOCKET_DIR)
    .filter(f => f.endsWith('.yml'))
    .map(f => path.join(DOCKET_DIR, f));
}

console.log(`🔧 Fixing docket file paths...\n`);

for (const docketFile of docketFiles) {
  const slug = path.basename(docketFile, '.yml');
  console.log(`\n📄 Processing: ${slug}`);
  
  let entries = [];
  try {
    entries = yaml.load(fs.readFileSync(docketFile, 'utf8')) || [];
  } catch (e) {
    console.error(`   ❌ Error reading YAML: ${e.message}`);
    continue;
  }
  
  let fixed = 0;
  
  for (const entry of entries) {
    if (!entry.file) continue;
    
    const originalFile = entry.file;
    let newFile = originalFile;
    
    // If path doesn't start with /, add it
    if (!newFile.startsWith('/')) {
      newFile = '/cases/' + slug + '/filings/' + newFile;
      fixed++;
    }
    // If path starts with / but doesn't have full path, fix it
    else if (!newFile.includes('/cases/')) {
      newFile = '/cases/' + slug + '/filings/' + path.basename(newFile);
      fixed++;
    }
    
    entry.file = newFile;
  }
  
  if (fixed > 0) {
    // Write back
    fs.writeFileSync(docketFile, yaml.dump(entries, { lineWidth: 1000 }));
    console.log(`   ✓ Fixed ${fixed} file path(s)`);
  } else {
    console.log(`   ✓ All paths already correct`);
  }
}

console.log(`\n✅ Done!\n`);

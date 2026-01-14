#!/usr/bin/env node

/**
 * Fix broken links and rename files for ATL-DC-007956-25
 * Problem: File named "proceeding-documents.pdf" should be "discovery-interrogatories-admissions.pdf"
 */

import fs from 'fs';
import path from 'path';
import yaml from 'js-yaml';

const CASE_SLUG = 'atl-dc-007956-25';
const DOCKET_DIR = `cases/${CASE_SLUG}/docket`;
const FILINGS_DIR = `cases/${CASE_SLUG}/filings`;
const DOCKET_YAML = `_data/docket/${CASE_SLUG}.yml`;

// File renames needed (old -> new)
const FILE_RENAMES = {
  '20251027-proceeding-documents.pdf': '20251027-discovery-interrogatories-admissions.pdf'
};

// YAML entry updates (id -> new file path)
const YAML_FIXES = {
  '2025-10-27-discovery-requests': {
    file: '20251027-discovery-interrogatories-admissions.pdf',
    title: 'Discovery Packet — Interrogatories, Notice to Produce, and Admissions'
  }
};

function renameFiles() {
  console.log('\n📝 Renaming files...\n');
  
  for (const [oldName, newName] of Object.entries(FILE_RENAMES)) {
    const oldDocketPath = path.join(DOCKET_DIR, oldName);
    const newDocketPath = path.join(DOCKET_DIR, newName);
    const oldFilingsPath = path.join(FILINGS_DIR, oldName);
    const newFilingsPath = path.join(FILINGS_DIR, newName);
    
    // Rename in docket directory
    if (fs.existsSync(oldDocketPath)) {
      fs.renameSync(oldDocketPath, newDocketPath);
      console.log(`  ✓ Renamed: ${oldDocketPath} → ${newDocketPath}`);
    }
    
    // Rename in filings directory if exists
    if (fs.existsSync(oldFilingsPath)) {
      fs.renameSync(oldFilingsPath, newFilingsPath);
      console.log(`  ✓ Renamed: ${oldFilingsPath} → ${newFilingsPath}`);
    }
  }
}

function fixYamlLinks() {
  console.log('\n🔧 Fixing YAML docket entries...\n');
  
  if (!fs.existsSync(DOCKET_YAML)) {
    console.error(`  ❌ Docket YAML not found: ${DOCKET_YAML}`);
    return;
  }
  
  const content = fs.readFileSync(DOCKET_YAML, 'utf8');
  const docket = yaml.load(content);
  
  let fixedCount = 0;
  
  for (const entry of docket) {
    if (!entry || !entry.id) continue;
    
    const fix = YAML_FIXES[entry.id];
    if (fix) {
      const oldFile = entry.file;
      entry.file = fix.file;
      if (fix.title) entry.title = fix.title;
      
      console.log(`  ✓ Fixed entry: ${entry.id}`);
      console.log(`    File: ${oldFile} → ${fix.file}`);
      if (fix.title) console.log(`    Title: ${fix.title}`);
      
      fixedCount++;
    }
  }
  
  if (fixedCount > 0) {
    const yamlOutput = yaml.dump(docket, {
      lineWidth: 1000,
      noRefs: true,
      sortKeys: false
    });
    
    fs.writeFileSync(DOCKET_YAML, yamlOutput, 'utf8');
    console.log(`\n  💾 Saved ${fixedCount} fix(es) to ${DOCKET_YAML}`);
  } else {
    console.log('  ℹ No YAML entries needed fixing');
  }
}

function verifyLinks() {
  console.log('\n✅ Verifying all links...\n');
  
  if (!fs.existsSync(DOCKET_YAML)) {
    console.error(`  ❌ Docket YAML not found: ${DOCKET_YAML}`);
    return;
  }
  
  const content = fs.readFileSync(DOCKET_YAML, 'utf8');
  const docket = yaml.load(content);
  
  let brokenCount = 0;
  let validCount = 0;
  
  for (const entry of docket) {
    if (!entry || !entry.file) continue;
    
    // Check in both docket and filings directories
    const docketPath = path.join(DOCKET_DIR, entry.file);
    const filingsPath = path.join(FILINGS_DIR, entry.file);
    
    const existsInDocket = fs.existsSync(docketPath);
    const existsInFilings = fs.existsSync(filingsPath);
    
    if (!existsInDocket && !existsInFilings) {
      console.log(`  ❌ BROKEN: ${entry.file} (${entry.title})`);
      brokenCount++;
    } else {
      validCount++;
    }
  }
  
  console.log(`\n📊 Verification Summary:`);
  console.log(`  Valid links: ${validCount}`);
  console.log(`  Broken links: ${brokenCount}`);
  
  if (brokenCount === 0) {
    console.log(`  ✅ All links are valid!`);
  }
}

function main() {
  console.log('='.repeat(70));
  console.log(`🔧 FIXING LINKS FOR ${CASE_SLUG.toUpperCase()}`);
  console.log('='.repeat(70));
  
  renameFiles();
  fixYamlLinks();
  verifyLinks();
  
  console.log('\n' + '='.repeat(70));
  console.log('✅ COMPLETE');
  console.log('='.repeat(70));
  console.log('\n💡 Next steps:');
  console.log('   1. Review changes: git diff');
  console.log('   2. Test build: bundle exec jekyll build');
  console.log('   3. Commit changes\n');
}

main();

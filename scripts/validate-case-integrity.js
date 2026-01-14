#!/usr/bin/env node
// Validate case folder integrity and docket YAML consistency
// Usage: node scripts/validate-case-integrity.js [case-slug]

import fs from 'fs';
import path from 'path';
import yaml from 'js-yaml';

const CASES_DIR = 'cases';
const DOCKET_DIR = '_data/docket';
const CASES_MAP_FILE = '_data/cases-map.yml';

const args = process.argv.slice(2);
const targetSlug = args[0];

// Load cases map
let casesMap = {};
if (fs.existsSync(CASES_MAP_FILE)) {
  casesMap = yaml.load(fs.readFileSync(CASES_MAP_FILE, 'utf8')) || {};
}

// Get all case slugs
const allSlugs = targetSlug ? [targetSlug] : [...new Set(Object.values(casesMap))];

console.log('🔍 Validating case integrity...\n');

let totalIssues = 0;

for (const slug of allSlugs) {
  console.log(`\n${'─'.repeat(60)}`);
  console.log(`📂 Case: ${slug}`);
  console.log('─'.repeat(60));
  
  const issues = [];
  
  // Check case folder structure
  const caseDir = path.join(CASES_DIR, slug);
  const filingsDir = path.join(caseDir, 'filings');
  const docketFile = path.join(DOCKET_DIR, `${slug}.yml`);
  
  if (!fs.existsSync(caseDir)) {
    issues.push(`❌ Case directory missing: ${caseDir}/`);
  }
  
  if (!fs.existsSync(filingsDir)) {
    issues.push(`❌ Filings directory missing: ${filingsDir}/`);
  }
  
  if (!fs.existsSync(docketFile)) {
    issues.push(`❌ Docket YAML missing: ${docketFile}`);
  }
  
  // Validate docket YAML
  if (fs.existsSync(docketFile)) {
    let docketEntries = [];
    try {
      docketEntries = yaml.load(fs.readFileSync(docketFile, 'utf8')) || [];
    } catch (e) {
      issues.push(`❌ Invalid YAML syntax in ${docketFile}: ${e.message}`);
    }
    
    // Check for duplicates
    const ids = docketEntries.map(e => e.id);
    const duplicateIds = ids.filter((id, idx) => ids.indexOf(id) !== idx);
    if (duplicateIds.length > 0) {
      issues.push(`⚠️  Duplicate docket IDs: ${[...new Set(duplicateIds)].join(', ')}`);
    }
    
    // Check for missing files
    const missingFiles = [];
    for (const entry of docketEntries) {
      if (!entry.file) continue;
      
      // Normalize file path
      let filePath = entry.file;
      if (filePath.startsWith('/')) filePath = filePath.slice(1);
      filePath = filePath.replace(/\\/g, '/');
      
      if (!fs.existsSync(filePath)) {
        missingFiles.push(entry.file);
      }
    }
    
    if (missingFiles.length > 0) {
      issues.push(`⚠️  Missing PDF files (${missingFiles.length}):`);
      missingFiles.slice(0, 5).forEach(f => {
        issues.push(`     ${f}`);
      });
      if (missingFiles.length > 5) {
        issues.push(`     ... and ${missingFiles.length - 5} more`);
      }
    }
    
    // Check for orphaned PDFs (files not in docket)
    if (fs.existsSync(filingsDir)) {
      const allPdfs = fs.readdirSync(filingsDir)
        .filter(f => f.toLowerCase().endsWith('.pdf'));
      
      const docketFiles = docketEntries
        .map(e => e.file ? path.basename(e.file) : null)
        .filter(Boolean);
      
      const orphaned = allPdfs.filter(pdf => !docketFiles.includes(pdf));
      
      if (orphaned.length > 0) {
        issues.push(`⚠️  Orphaned PDFs (not in docket YAML): ${orphaned.length} files`);
        orphaned.slice(0, 3).forEach(f => {
          issues.push(`     ${f}`);
        });
        if (orphaned.length > 3) {
          issues.push(`     ... and ${orphaned.length - 3} more`);
        }
      }
    }
    
    console.log(`   Docket entries: ${docketEntries.length}`);
  }
  
  // Check for files
  if (fs.existsSync(filingsDir)) {
    const pdfCount = fs.readdirSync(filingsDir)
      .filter(f => f.toLowerCase().endsWith('.pdf')).length;
    console.log(`   PDF files: ${pdfCount}`);
  }
  
  // Report issues
  if (issues.length === 0) {
    console.log(`\n   ✅ No issues found - case integrity looks good!`);
  } else {
    console.log(`\n   Issues found (${issues.length}):\n`);
    issues.forEach(issue => console.log(`   ${issue}`));
    totalIssues += issues.length;
  }
}

console.log(`\n${'='.repeat(60)}`);
if (totalIssues === 0) {
  console.log('✅ All cases validated successfully!');
} else {
  console.log(`⚠️  Total issues found: ${totalIssues}`);
  console.log('\nRecommended actions:');
  console.log('  - Fix duplicate IDs in docket YAML files');
  console.log('  - Add missing PDFs or remove stale docket entries');
  console.log('  - Add orphaned PDFs to docket YAML with proper metadata');
}
console.log('='.repeat(60) + '\n');

process.exit(totalIssues > 0 ? 1 : 0);

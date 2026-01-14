#!/usr/bin/env node

/**
 * Fix all docket file paths to use absolute URLs
 * Problem: YAML files use relative paths, need /cases/<slug>/filings/<file>
 */

import fs from 'fs';
import path from 'path';
import yaml from 'js-yaml';

const DOCKET_DIR = '_data/docket';
const CASES_DIR = 'cases';

function fixDocketPaths(slug) {
  const yamlPath = path.join(DOCKET_DIR, `${slug}.yml`);
  
  if (!fs.existsSync(yamlPath)) {
    return { slug, fixed: 0, skipped: true };
  }
  
  const content = fs.readFileSync(yamlPath, 'utf8');
  const docket = yaml.load(content);
  
  if (!Array.isArray(docket)) {
    return { slug, fixed: 0, skipped: true };
  }
  
  let fixedCount = 0;
  
  for (const entry of docket) {
    if (!entry || !entry.file) continue;
    
    const file = entry.file;
    
    // Skip if already has absolute path
    if (file.startsWith('/cases/') || file.startsWith('/assets/')) {
      continue;
    }
    
    // Skip if it's a bare filename without path
    if (!file.includes('/')) {
      // Check if file exists in docket or filings directory
      const docketPath = path.join(CASES_DIR, slug, 'docket', file);
      const filingsPath = path.join(CASES_DIR, slug, 'filings', file);
      
      let correctPath = null;
      
      if (fs.existsSync(docketPath)) {
        correctPath = `/cases/${slug}/docket/${file}`;
      } else if (fs.existsSync(filingsPath)) {
        correctPath = `/cases/${slug}/filings/${file}`;
      } else {
        console.log(`  ⚠️  File not found: ${file} (${entry.title})`);
        continue;
      }
      
      entry.file = correctPath;
      fixedCount++;
    }
  }
  
  if (fixedCount > 0) {
    const yamlOutput = yaml.dump(docket, {
      lineWidth: 1000,
      noRefs: true,
      sortKeys: false
    });
    
    fs.writeFileSync(yamlPath, yamlOutput, 'utf8');
  }
  
  return { slug, fixed: fixedCount, skipped: false };
}

function main() {
  console.log('='.repeat(70));
  console.log('🔧 FIXING ALL DOCKET FILE PATHS');
  console.log('='.repeat(70));
  console.log('Converting relative paths to absolute URLs...\n');
  
  if (!fs.existsSync(DOCKET_DIR)) {
    console.error(`❌ Docket directory not found: ${DOCKET_DIR}`);
    process.exit(1);
  }
  
  const yamlFiles = fs.readdirSync(DOCKET_DIR).filter(f => f.endsWith('.yml'));
  
  const results = [];
  
  for (const yamlFile of yamlFiles) {
    const slug = yamlFile.replace('.yml', '');
    console.log(`\n📄 Processing: ${slug}`);
    
    const result = fixDocketPaths(slug);
    results.push(result);
    
    if (result.skipped) {
      console.log('  ⊘ Skipped (empty or invalid)');
    } else if (result.fixed === 0) {
      console.log('  ✓ All paths already absolute');
    } else {
      console.log(`  ✓ Fixed ${result.fixed} path(s)`);
    }
  }
  
  const totalFixed = results.reduce((sum, r) => sum + r.fixed, 0);
  const filesChanged = results.filter(r => r.fixed > 0).length;
  
  console.log('\n' + '='.repeat(70));
  console.log('📊 SUMMARY');
  console.log('='.repeat(70));
  console.log(`Files processed: ${yamlFiles.length}`);
  console.log(`Files changed: ${filesChanged}`);
  console.log(`Paths fixed: ${totalFixed}`);
  
  if (filesChanged > 0) {
    console.log('\n💡 Next steps:');
    console.log('   1. Review: git diff _data/docket/');
    console.log('   2. Verify: node scripts/check-pdf-links.js');
    console.log('   3. Commit changes');
  }
  
  console.log('');
}

main();

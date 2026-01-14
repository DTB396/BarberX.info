#!/usr/bin/env node

/**
 * Optimize docket entry titles by removing redundant dates and normalizing formatting
 * Problem: Recent intake creates titles like "2025 12 27 Filing 2025 12 27 certification..."
 * Solution: Clean to just "Certification..." (date already in date field)
 */

import fs from 'fs';
import path from 'path';
import yaml from 'js-yaml';

const DOCKET_DIR = '_data/docket';

function cleanTitle(title, type, date) {
  if (!title || typeof title !== 'string') return title;
  
  let cleaned = title;
  
  // Remove date prefixes (YYYY MM DD format with spaces, dashes, or underscores)
  cleaned = cleaned.replace(/^\d{4}[\s\-_]+\d{1,2}[\s\-_]+\d{1,2}[\s\-_]+/g, '');
  
  // Remove generic "Filing " prefix (most common redundancy)
  cleaned = cleaned.replace(/^Filing[\s\-_]+/i, '');
  
  // Remove any remaining date patterns in middle of title
  cleaned = cleaned.replace(/\d{4}[\s\-_]+\d{1,2}[\s\-_]+\d{1,2}[\s\-_]+/g, '');
  
  // Normalize separators: convert underscores to spaces
  cleaned = cleaned.replace(/[_]+/g, ' ');
  
  // Clean up multiple spaces
  cleaned = cleaned.replace(/\s+/g, ' ').trim();
  
  // Capitalize first letter
  if (cleaned.length > 0) {
    cleaned = cleaned.charAt(0).toUpperCase() + cleaned.slice(1);
  }
  
  return cleaned || title; // Return original if cleaning resulted in empty string
}

function optimizeDocketFile(filepath) {
  console.log(`\n📄 Processing: ${path.basename(filepath)}`);
  
  const content = fs.readFileSync(filepath, 'utf8');
  const docket = yaml.load(content);
  
  if (!Array.isArray(docket) || docket.length === 0) {
    console.log('   ⊘ Empty or invalid docket file');
    return 0;
  }
  
  let optimizedCount = 0;
  
  for (const entry of docket) {
    if (!entry || !entry.title) continue;
    
    const original = entry.title;
    const optimized = cleanTitle(entry.title, entry.type, entry.date);
    
    if (original !== optimized) {
      console.log(`   ✓ ${original}`);
      console.log(`     → ${optimized}`);
      entry.title = optimized;
      optimizedCount++;
    }
  }
  
  if (optimizedCount > 0) {
    // Write back with consistent formatting
    const yamlOutput = yaml.dump(docket, { 
      lineWidth: 1000,
      noRefs: true,
      sortKeys: false
    });
    
    fs.writeFileSync(filepath, yamlOutput, 'utf8');
    console.log(`   💾 Saved ${optimizedCount} optimized title(s)`);
  } else {
    console.log('   ✓ All titles already optimized');
  }
  
  return optimizedCount;
}

function main() {
  console.log('='.repeat(70));
  console.log('🔧 DOCKET TITLE OPTIMIZER');
  console.log('='.repeat(70));
  console.log('Removing redundant dates and normalizing title formatting...\n');
  
  if (!fs.existsSync(DOCKET_DIR)) {
    console.error(`❌ Docket directory not found: ${DOCKET_DIR}`);
    process.exit(1);
  }
  
  const files = fs.readdirSync(DOCKET_DIR)
    .filter(f => f.endsWith('.yml'))
    .map(f => path.join(DOCKET_DIR, f));
  
  if (files.length === 0) {
    console.log('⊘ No docket files found');
    return;
  }
  
  console.log(`Found ${files.length} docket file(s)\n`);
  
  let totalOptimized = 0;
  let filesChanged = 0;
  
  for (const filepath of files) {
    const count = optimizeDocketFile(filepath);
    totalOptimized += count;
    if (count > 0) filesChanged++;
  }
  
  console.log('\n' + '='.repeat(70));
  console.log('📊 SUMMARY');
  console.log('='.repeat(70));
  console.log(`Files processed: ${files.length}`);
  console.log(`Files changed: ${filesChanged}`);
  console.log(`Titles optimized: ${totalOptimized}`);
  
  if (filesChanged > 0) {
    console.log('\n💡 Next steps:');
    console.log('   1. Review changes: git diff _data/docket/');
    console.log('   2. Test build: bundle exec jekyll build');
    console.log('   3. Commit: git add _data/docket/ && git commit -m "Optimize docket titles"');
  }
  
  console.log('');
}

main();

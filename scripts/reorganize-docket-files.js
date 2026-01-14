#!/usr/bin/env node
// Reorganize docket files from cases/<slug>/filings/ to assets/cases/<slug>/docket/
// and update all YAML docket files with correct paths

import fs from 'fs';
import path from 'path';
import yaml from 'js-yaml';

const CASES_DIR = 'cases';
const ASSETS_CASES_DIR = 'assets/cases';
const DOCKET_DATA_DIR = '_data/docket';

function ensureDir(dirPath) {
  fs.mkdirSync(dirPath, { recursive: true });
}

function copyFile(src, dest) {
  ensureDir(path.dirname(dest));
  fs.copyFileSync(src, dest);
  console.log(`  ✓ Copied: ${src} → ${dest}`);
}

function updateDocketYaml(slug, oldPath, newPath) {
  const yamlPath = path.join(DOCKET_DATA_DIR, `${slug}.yml`);
  
  if (!fs.existsSync(yamlPath)) {
    console.log(`  ⚠ No docket YAML found for ${slug}`);
    return;
  }

  let content = fs.readFileSync(yamlPath, 'utf8');
  const originalContent = content;
  
  // Update the path
  content = content.replace(oldPath, newPath);
  
  if (content !== originalContent) {
    fs.writeFileSync(yamlPath, content);
    console.log(`  ✓ Updated ${yamlPath}: ${oldPath} → ${newPath}`);
  }
}

function main() {
  console.log('=== Reorganizing Docket Files ===\n');
  
  // Ensure assets/cases directory exists
  ensureDir(ASSETS_CASES_DIR);
  
  // Get all case directories
  const caseDirs = fs.readdirSync(CASES_DIR).filter(name => {
    const fullPath = path.join(CASES_DIR, name);
    return fs.statSync(fullPath).isDirectory();
  });
  
  let totalFiles = 0;
  let totalUpdates = 0;
  
  for (const slug of caseDirs) {
    const filingsDir = path.join(CASES_DIR, slug, 'filings');
    
    if (!fs.existsSync(filingsDir)) {
      console.log(`⊘ No filings directory for ${slug}`);
      continue;
    }
    
    console.log(`\n📁 Processing ${slug}...`);
    
    const docketDir = path.join(ASSETS_CASES_DIR, slug, 'docket');
    ensureDir(docketDir);
    
    // Get all PDFs in filings directory
    const files = fs.readdirSync(filingsDir).filter(f => f.endsWith('.pdf'));
    
    if (files.length === 0) {
      console.log(`  ⊘ No PDF files found`);
      continue;
    }
    
    for (const filename of files) {
      const srcPath = path.join(filingsDir, filename);
      const destPath = path.join(docketDir, filename);
      
      // Copy file to new location
      copyFile(srcPath, destPath);
      totalFiles++;
      
      // Update docket YAML
      const oldUrlPath = `/cases/${slug}/filings/${filename}`;
      const newUrlPath = `/assets/cases/${slug}/docket/${filename}`;
      
      // Also try variants that might exist in the YAML
      const oldVariants = [
        `/cases/${slug}/${filename}`,
        `/cases/${slug}/pcr/${filename}`,
        `/cases/${slug}/docket/${filename}`,
        `/cases/${slug}/filings/${filename}`
      ];
      
      for (const oldPath of oldVariants) {
        updateDocketYaml(slug, oldPath, newUrlPath);
      }
      
      totalUpdates++;
    }
    
    console.log(`  ✓ Processed ${files.length} files`);
  }
  
  console.log(`\n=== Summary ===`);
  console.log(`Total files copied: ${totalFiles}`);
  console.log(`Total YAML entries updated: ${totalUpdates}`);
  console.log(`\nNext steps:`);
  console.log(`1. Review the changes with: git status`);
  console.log(`2. Test Jekyll build: bundle exec jekyll build`);
  console.log(`3. Verify case pages display correctly`);
  console.log(`4. If everything looks good, you can remove the old filings directories`);
}

main();

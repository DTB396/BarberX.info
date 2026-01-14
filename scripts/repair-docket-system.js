#!/usr/bin/env node
/**
 * FaithFrontier Docket System Repair Tool
 * 
 * This script fixes the docket system by:
 * 1. Copying PDFs from cases/<slug>/filings/ to assets/cases/<slug>/docket/
 * 2. Updating _data/docket/*.yml files to reference the correct paths
 * 3. Matching actual filenames on disk (which use YYYYMMDD-name.pdf format)
 * 
 * Run this script from the repository root:
 *   node scripts/repair-docket-system.js
 */

import fs from 'fs';
import path from 'path';
import yaml from 'js-yaml';

const CASES_DIR = 'cases';
const ASSETS_CASES_DIR = 'assets/cases';
const DOCKET_DATA_DIR = '_data/docket';

// Normalize filename for matching
function normalizeFilename(filename) {
  return filename
    .toLowerCase()
    .replace(/[\s_]+/g, '-')
    .replace(/\.pdf$/i, '');
}

// Find matching file in directory
function findMatchingFile(targetName, filesInDir) {
  const normalizedTarget = normalizeFilename(targetName);
  
  for (const file of filesInDir) {
    const normalized = normalizeFilename(file);
    if (normalized.includes(normalizedTarget) || normalizedTarget.includes(normalized)) {
      return file;
    }
  }
  
  // Try date-based matching (YYYY-MM-DD vs YYYYMMDD)
  const dateMatch = targetName.match(/(\d{4})-(\d{2})-(\d{2})/);
  if (dateMatch) {
    const compactDate = `${dateMatch[1]}${dateMatch[2]}${dateMatch[3]}`;
    for (const file of filesInDir) {
      if (file.startsWith(compactDate)) {
        const targetSuffix = targetName.replace(/^\d{4}-\d{2}-\d{2}[_-]?/, '').toLowerCase();
        const fileSuffix = file.replace(/^\d{8}-/, '').toLowerCase();
        if (fileSuffix.includes(targetSuffix.substring(0, 20)) || 
            targetSuffix.includes(fileSuffix.substring(0, 20))) {
          return file;
        }
      }
    }
  }
  
  return null;
}

// Ensure directory exists
function ensureDir(dirPath) {
  fs.mkdirSync(dirPath, { recursive: true });
}

// Copy files and track mappings
function copyAndMapFiles() {
  const mappings = {};
  let totalCopied = 0;
  let totalErrors = 0;
  
  const caseDirs = fs.readdirSync(CASES_DIR).filter(name => {
    const fullPath = path.join(CASES_DIR, name);
    return fs.statSync(fullPath).isDirectory();
  });
  
  for (const slug of caseDirs) {
    const filingsDir = path.join(CASES_DIR, slug, 'filings');
    if (!fs.existsSync(filingsDir)) {
      console.log(`⊘ No filings directory for ${slug}`);
      continue;
    }
    
    console.log(`\n📁 Processing ${slug}...`);
    
    const docketDir = path.join(ASSETS_CASES_DIR, slug, 'docket');
    ensureDir(docketDir);
    
    const files = fs.readdirSync(filingsDir).filter(f => f.toLowerCase().endsWith('.pdf'));
    
    if (files.length === 0) {
      console.log(`  ⊘ No PDF files found`);
      continue;
    }
    
    mappings[slug] = {};
    
    for (const filename of files) {
      const srcPath = path.join(filingsDir, filename);
      const destPath = path.join(docketDir, filename);
      
      try {
        if (!fs.existsSync(destPath)) {
          fs.copyFileSync(srcPath, destPath);
          console.log(`  ✓ Copied: ${filename}`);
          totalCopied++;
        } else {
          console.log(`  → Exists: ${filename}`);
        }
        
        // Track the filename mapping
        mappings[slug][filename] = filename;
        
      } catch (error) {
        console.log(`  ✗ Error: ${filename} - ${error.message}`);
        totalErrors++;
      }
    }
    
    console.log(`  📊 ${slug}: ${files.length} files processed`);
  }
  
  console.log(`\n✅ Copy complete: ${totalCopied} files copied, ${totalErrors} errors`);
  return mappings;
}

// Update YAML files with correct paths
function updateYAMLFiles(fileMappings) {
  let totalUpdated = 0;
  let totalPaths = 0;
  
  console.log(`\n📝 Updating YAML files...`);
  
  const yamlFiles = fs.readdirSync(DOCKET_DATA_DIR).filter(f => f.endsWith('.yml'));
  
  for (const yamlFile of yamlFiles) {
    const slug = path.basename(yamlFile, '.yml');
    const yamlPath = path.join(DOCKET_DATA_DIR, yamlFile);
    
    console.log(`\n  Updating ${yamlFile}...`);
    
    let content = fs.readFileSync(yamlPath, 'utf8');
    const originalContent = content;
    let pathsUpdated = 0;
    
    // Get list of actual files for this case
    const docketDir = path.join(ASSETS_CASES_DIR, slug, 'docket');
    let actualFiles = [];
    if (fs.existsSync(docketDir)) {
      actualFiles = fs.readdirSync(docketDir).filter(f => f.toLowerCase().endsWith('.pdf'));
    }
    
    // Parse YAML to update each entry
    try {
      const entries = yaml.load(content) || [];
      
      for (const entry of entries) {
        if (!entry.file) continue;
        
        const oldPath = entry.file;
        
        // Extract filename from old path
        let filename = path.basename(oldPath);
        
        // Try to find matching actual file
        const matchedFile = findMatchingFile(filename, actualFiles);
        if (matchedFile) {
          filename = matchedFile;
        }
        
        // Create new path
        const newPath = `/assets/cases/${slug}/docket/${filename}`;
        
        if (oldPath !== newPath) {
          // Replace in content (preserve exact formatting)
          content = content.replace(
            new RegExp(`file:\\s*${oldPath.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}`),
            `file: ${newPath}`
          );
          pathsUpdated++;
        }
      }
      
    } catch (error) {
      console.log(`    ⚠ Error parsing YAML: ${error.message}`);
      console.log(`    Attempting regex-based update...`);
      
      // Fallback: regex replacement
      const patterns = [
        [new RegExp(`/cases/${slug}/[^"\\s]+\\.pdf`, 'g'), (match) => {
          const filename = path.basename(match);
          return `/assets/cases/${slug}/docket/${filename}`;
        }],
        [new RegExp(`/cases/atl-24-001934/pcr/[^"\\s]+\\.pdf`, 'g'), (match) => {
          const filename = path.basename(match);
          return `/assets/cases/${slug}/docket/${filename}`;
        }]
      ];
      
      for (const [pattern, replacer] of patterns) {
        const newContent = content.replace(pattern, replacer);
        if (newContent !== content) {
          const matches = (content.match(pattern) || []).length;
          pathsUpdated += matches;
          content = newContent;
        }
      }
    }
    
    if (content !== originalContent) {
      fs.writeFileSync(yamlPath, content);
      console.log(`    ✓ Updated ${pathsUpdated} paths`);
      totalUpdated++;
      totalPaths += pathsUpdated;
    } else {
      console.log(`    → No changes needed`);
    }
  }
  
  console.log(`\n✅ YAML update complete: ${totalUpdated} files updated, ${totalPaths} paths fixed`);
}

// Main function
function main() {
  console.log('='.repeat(60));
  console.log('FaithFrontier Docket System Repair Tool');
  console.log('='.repeat(60));
  console.log();
  
  // Step 1: Copy files
  console.log('STEP 1: Copying PDF files to new location');
  console.log('-'.repeat(60));
  const mappings = copyAndMapFiles();
  
  // Step 2: Update YAML files
  console.log('\n' + '='.repeat(60));
  console.log('STEP 2: Updating YAML docket files');
  console.log('-'.repeat(60));
  updateYAMLFiles(mappings);
  
  // Summary
  console.log('\n' + '='.repeat(60));
  console.log('✅ REPAIR COMPLETE!');
  console.log('='.repeat(60));
  console.log('\nNext steps:');
  console.log('1. Review changes: git status');
  console.log('2. Test build: bundle exec jekyll build');
  console.log('3. Check /cases/ page locally: bundle exec jekyll serve');
  console.log('4. Commit if everything looks good');
  console.log();
}

// Run
try {
  main();
} catch (error) {
  console.error('\n❌ Fatal error:', error.message);
  console.error(error.stack);
  process.exit(1);
}

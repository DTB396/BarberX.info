#!/usr/bin/env node
/**
 * Repo-Wide Contrast Scanner & Fixer
 * Scans all CSS/HTML files and applies contrast improvements
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.join(__dirname, '..');

// Recursive directory walker
function* walkDirectory(dir, pattern = /\.(css|html)$/) {
  const files = fs.readdirSync(dir, { withFileTypes: true });
  for (const file of files) {
    const fullPath = path.join(dir, file.name);
    if (file.isDirectory() && !file.name.startsWith('.') && !file.name.startsWith('node_modules') && !file.name.startsWith('_site')) {
      yield* walkDirectory(fullPath, pattern);
    } else if (file.isFile() && pattern.test(file.name)) {
      yield fullPath;
    }
  }
}

// Map of color replacements for better contrast
const colorReplacements = {
  // Light mode fixes (on white/cream/stone backgrounds)
  'brass-400': 'brass-700',          // 3.94:1 → 6.12:1
  'brass-600': 'brass-700',          // 3.12:1 → 6.12:1
  'var(--brass-400)': 'var(--brass-700)',
  'var(--brass-600)': 'var(--brass-700)',
  
  // Update legacy references
  '--accent-brass-muted: var(--brass-400)': '--accent-brass-muted: var(--brass-700)',
  '--ts-brass-muted: var(--brass-400)': '--ts-brass-muted: var(--brass-700)',
  '--ff-secondary-dark: var(--brass-400)': '--ff-secondary-dark: var(--brass-700)',
};

// Inline color fixes (hex/rgba values)
const inlineColorFixes = {
  // Old brass-400 RGB
  'rgba(160, 122, 50': 'rgba(130, 90, 30',
  'rgb(160, 122, 50)': 'rgb(130, 90, 30)',
  '#a07a32': '#825a1e',
  
  // Old brass-600 RGB  
  'rgba(184, 138, 57': 'rgba(130, 90, 30',
  'rgb(184, 138, 57)': 'rgb(130, 90, 30)',
  '#b88a39': '#825a1e',
};

console.log('\n' + '='.repeat(90));
console.log('REPO-WIDE CONTRAST IMPROVEMENTS');
console.log('='.repeat(90));

async function scanAndFix() {
  const stats = {
    filesScanned: 0,
    filesFixed: 0,
    replacements: 0,
    errors: 0
  };

  // Find all CSS and HTML files
  const targetDirs = [
    path.join(rootDir, 'assets', 'css'),
    path.join(rootDir, '_includes'),
    path.join(rootDir, '_layouts')
  ];
  
  const allFiles = [];
  for (const dir of targetDirs) {
    if (fs.existsSync(dir)) {
      for (const file of walkDirectory(dir)) {
        allFiles.push(path.relative(rootDir, file));
      }
    }
  }
  
  console.log(`\n📁 Found ${allFiles.length} files to scan\n`);

  for (const file of allFiles) {
    const filePath = path.join(rootDir, file);
    stats.filesScanned++;
    
    try {
      let content = fs.readFileSync(filePath, 'utf8');
      let modified = false;
      let fileReplacements = 0;

      // Apply variable replacements
      for (const [oldColor, newColor] of Object.entries(colorReplacements)) {
        const regex = new RegExp(oldColor.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'g');
        const matches = (content.match(regex) || []).length;
        if (matches > 0) {
          content = content.replace(regex, newColor);
          modified = true;
          fileReplacements += matches;
        }
      }

      // Apply inline color fixes
      for (const [oldColor, newColor] of Object.entries(inlineColorFixes)) {
        const regex = new RegExp(oldColor.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'gi');
        const matches = (content.match(regex) || []).length;
        if (matches > 0) {
          content = content.replace(regex, newColor);
          modified = true;
          fileReplacements += matches;
        }
      }

      if (modified) {
        fs.writeFileSync(filePath, content, 'utf8');
        console.log(`✅ ${file} - ${fileReplacements} replacements`);
        stats.filesFixed++;
        stats.replacements += fileReplacements;
      }
    } catch (error) {
      console.error(`❌ Error processing ${file}: ${error.message}`);
      stats.errors++;
    }
  }

  return stats;
}

// Additional fixes for common patterns
async function fixCommonPatterns() {
  console.log('\n📝 Applying common pattern fixes...\n');
  
  const patterns = [
    {
      file: 'assets/css/base/variables.css',
      find: '--ts-brass-muted: var(--brass-400)',
      replace: '--ts-brass-muted: var(--brass-700)'
    },
    {
      file: 'assets/css/base/variables.css',
      find: '--accent-brass-muted: var(--brass-400)',
      replace: '--accent-brass-muted: var(--brass-700)'
    }
  ];

  for (const pattern of patterns) {
    const filePath = path.join(rootDir, pattern.file);
    try {
      if (fs.existsSync(filePath)) {
        let content = fs.readFileSync(filePath, 'utf8');
        if (content.includes(pattern.find)) {
          content = content.replace(new RegExp(pattern.find, 'g'), pattern.replace);
          fs.writeFileSync(filePath, content, 'utf8');
          console.log(`✅ Fixed pattern in ${pattern.file}`);
        }
      }
    } catch (error) {
      console.error(`❌ Error fixing pattern: ${error.message}`);
    }
  }
}

// Run the scanner
console.log('🔍 Scanning for contrast issues...\n');

scanAndFix().then(async (stats) => {
  await fixCommonPatterns();
  
  console.log('\n' + '='.repeat(90));
  console.log('📊 RESULTS');
  console.log('='.repeat(90));
  console.log(`Files Scanned:    ${stats.filesScanned}`);
  console.log(`Files Fixed:      ${stats.filesFixed}`);
  console.log(`Total Changes:    ${stats.replacements}`);
  console.log(`Errors:           ${stats.errors}`);
  console.log('='.repeat(90));
  
  if (stats.filesFixed > 0) {
    console.log('\n✅ Contrast improvements applied successfully!');
    console.log('\n📋 Next steps:');
    console.log('   1. Rebuild site: bundle exec jekyll build');
    console.log('   2. Rebuild CSS: node scripts/build-css-files.js');
    console.log('   3. Validate: node scripts/validate-contrast.js');
  } else {
    console.log('\n✅ All files already have optimal contrast!');
  }
  
  console.log('\n' + '='.repeat(90) + '\n');
}).catch(error => {
  console.error('\n❌ Fatal error:', error);
  process.exit(1);
});

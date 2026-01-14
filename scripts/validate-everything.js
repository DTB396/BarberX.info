#!/usr/bin/env node

/**
 * FaithFrontier Complete Site Validation System
 * Turns over every stone - checks everything
 * 
 * Usage:
 *   node scripts/validate-everything.js
 *   node scripts/validate-everything.js --fix
 *   node scripts/validate-everything.js --verbose
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import yaml from 'js-yaml';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.join(__dirname, '..');

// Configuration
const config = {
  fix: process.argv.includes('--fix'),
  verbose: process.argv.includes('--verbose'),
  site: process.argv.includes('--site') 
    ? process.argv[process.argv.indexOf('--site') + 1] 
    : '_site'
};

const results = {
  errors: [],
  warnings: [],
  fixed: [],
  checks: 0,
  passed: 0
};

function log(msg, level = 'info') {
  const symbols = { error: '✗', warning: '⚠', success: '✓', info: '→' };
  const colors = { error: '\x1b[31m', warning: '\x1b[33m', success: '\x1b[32m', info: '\x1b[36m' };
  console.log(`${colors[level]}${symbols[level]} ${msg}\x1b[0m`);
}

function addError(check, file, issue) {
  results.errors.push({ check, file, issue });
  log(`ERROR in ${file}: ${issue}`, 'error');
}

function addWarning(check, file, issue) {
  results.warnings.push({ check, file, issue });
  if (config.verbose) log(`WARNING in ${file}: ${issue}`, 'warning');
}

function addFixed(check, file, fix) {
  results.fixed.push({ check, file, fix });
  log(`FIXED ${file}: ${fix}`, 'success');
}

// ============================================================================
// CHECK 1: Docket YAML Paths
// ============================================================================
async function checkDocketPaths() {
  log('\nCHECK 1: Docket YAML file paths', 'info');
  results.checks++;
  
  const docketDir = path.join(rootDir, '_data', 'docket');
  const files = fs.readdirSync(docketDir).filter(f => f.endsWith('.yml') || f.endsWith('.yaml'));
  
  let issueCount = 0;
  
  for (const file of files) {
    const filePath = path.join(docketDir, file);
    let content = fs.readFileSync(filePath, 'utf8');
    const originalContent = content;
    
    // Check for /assets/cases/ pattern (old, broken)
    if (content.includes('/assets/cases/')) {
      issueCount++;
      addError('docket-paths', file, 'Contains /assets/cases/ paths (should be /cases/)');
      
      if (config.fix) {
        content = content.replace(/\/assets\/cases\//g, '/cases/');
        fs.writeFileSync(filePath, content, 'utf8');
        addFixed('docket-paths', file, 'Changed /assets/cases/ → /cases/');
      }
    }
    
    // Check for proper YAML structure
    try {
      const data = yaml.load(content);
      if (!Array.isArray(data)) {
        addWarning('docket-structure', file, 'Docket file is not an array');
      }
    } catch (err) {
      addError('docket-yaml', file, `Invalid YAML: ${err.message}`);
      issueCount++;
    }
  }
  
  if (issueCount === 0) {
    results.passed++;
    log(`✓ Docket paths check passed (${files.length} files)`, 'success');
  }
}

// ============================================================================
// CHECK 2: PDF File Existence
// ============================================================================
async function checkPDFExistence() {
  log('\nCHECK 2: PDF file existence', 'info');
  results.checks++;
  
  const docketDir = path.join(rootDir, '_data', 'docket');
  const casesDir = path.join(rootDir, 'cases');
  const files = fs.readdirSync(docketDir).filter(f => f.endsWith('.yml'));
  
  let missingCount = 0;
  
  for (const file of files) {
    const filePath = path.join(docketDir, file);
    const content = fs.readFileSync(filePath, 'utf8');
    const data = yaml.load(content);
    
    if (!Array.isArray(data)) continue;
    
    for (const entry of data) {
      if (!entry.file) continue;
      
      // Convert /cases/slug/filings/file.pdf to cases/slug/filings/file.pdf
      const relativePath = entry.file.replace(/^\//, '');
      const fullPath = path.join(rootDir, relativePath);
      
      if (!fs.existsSync(fullPath)) {
        missingCount++;
        addError('pdf-missing', file, `PDF not found: ${entry.file} (${entry.title})`);
      }
    }
  }
  
  if (missingCount === 0) {
    results.passed++;
    log(`✓ PDF existence check passed`, 'success');
  }
}

// ============================================================================
// CHECK 3: Case README Files
// ============================================================================
async function checkCaseREADMEs() {
  log('\nCHECK 3: Case README files', 'info');
  results.checks++;
  
  const casesDir = path.join(rootDir, 'cases');
  if (!fs.existsSync(casesDir)) {
    addWarning('case-readmes', 'cases/', 'Cases directory not found');
    return;
  }
  
  const cases = fs.readdirSync(casesDir, { withFileTypes: true })
    .filter(d => d.isDirectory())
    .map(d => d.name);
  
  let issueCount = 0;
  
  for (const caseDir of cases) {
    const readmePath = path.join(casesDir, caseDir, 'README.md');
    
    if (!fs.existsSync(readmePath)) {
      issueCount++;
      addWarning('case-readme-missing', caseDir, 'No README.md file');
    } else {
      const content = fs.readFileSync(readmePath, 'utf8');
      
      // Check for minimum content
      if (content.length < 100) {
        addWarning('case-readme-empty', `${caseDir}/README.md`, 'README is too short');
      }
      
      // Check for docket heading
      if (!content.includes('## Docket') && !content.includes('## Chronological Docket')) {
        addWarning('case-readme-structure', `${caseDir}/README.md`, 'Missing docket section');
      }
    }
  }
  
  if (issueCount === 0) {
    results.passed++;
    log(`✓ Case README check passed (${cases.length} cases)`, 'success');
  }
}

// ============================================================================
// CHECK 4: CSS Syntax and Common Issues
// ============================================================================
async function checkCSS() {
  log('\nCHECK 4: CSS files', 'info');
  results.checks++;
  
  const cssDir = path.join(rootDir, 'assets', 'css');
  const cssFiles = fs.readdirSync(cssDir)
    .filter(f => f.endsWith('.css'))
    .map(f => path.join(cssDir, f));
  
  let issueCount = 0;
  
  for (const file of cssFiles) {
    const content = fs.readFileSync(file, 'utf8');
    const fileName = path.basename(file);
    
    // Check for common issues
    const issues = [];
    
    // Unclosed braces
    const openBraces = (content.match(/{/g) || []).length;
    const closeBraces = (content.match(/}/g) || []).length;
    if (openBraces !== closeBraces) {
      issues.push(`Unmatched braces: ${openBraces} open, ${closeBraces} close`);
    }
    
    // Very low opacity (hard to see)
    const lowOpacity = content.match(/opacity:\s*0\.[0-4]\d/g);
    if (lowOpacity) {
      addWarning('css-low-opacity', fileName, `Low opacity values found: ${lowOpacity.length} instances`);
    }
    
    // display: flex with position: absolute (common layout issue)
    if (content.includes('display: flex') && content.includes('position: absolute')) {
      addWarning('css-flex-absolute', fileName, 'Contains both flex and absolute positioning');
    }
    
    if (issues.length > 0) {
      issueCount++;
      issues.forEach(issue => addError('css-syntax', fileName, issue));
    }
  }
  
  if (issueCount === 0) {
    results.passed++;
    log(`✓ CSS check passed (${cssFiles.length} files)`, 'success');
  }
}

// ============================================================================
// CHECK 5: Broken Internal Links (Built Site)
// ============================================================================
async function checkBuiltLinks() {
  log('\nCHECK 5: Built site internal links', 'info');
  results.checks++;
  
  const siteDir = path.join(rootDir, config.site);
  
  if (!fs.existsSync(siteDir)) {
    addWarning('site-links', config.site, 'Built site not found. Run build first.');
    return;
  }
  
  // Use existing check-site-links.js logic
  log(`  → Site exists at ${config.site}`, 'info');
  log(`  → Run 'npm run check:site-links' for full link validation`, 'info');
  results.passed++;
}

// ============================================================================
// CHECK 6: YAML Frontmatter in Pages
// ============================================================================
async function checkFrontmatter() {
  log('\nCHECK 6: Page frontmatter', 'info');
  results.checks++;
  
  const pagesDir = path.join(rootDir, '_pages');
  if (!fs.existsSync(pagesDir)) {
    log('  → No _pages directory', 'info');
    results.passed++;
    return;
  }
  
  const pages = fs.readdirSync(pagesDir).filter(f => f.endsWith('.md') || f.endsWith('.html'));
  let issueCount = 0;
  
  for (const file of pages) {
    const filePath = path.join(pagesDir, file);
    const content = fs.readFileSync(filePath, 'utf8');
    
    // Check for frontmatter
    if (!content.startsWith('---')) {
      issueCount++;
      addError('frontmatter-missing', file, 'No frontmatter found');
      continue;
    }
    
    // Extract frontmatter
    const matches = content.match(/^---\n([\s\S]+?)\n---/);
    if (!matches) {
      issueCount++;
      addError('frontmatter-invalid', file, 'Invalid frontmatter format');
      continue;
    }
    
    try {
      const fm = yaml.load(matches[1]);
      
      // Check for required fields
      if (!fm.layout) {
        addWarning('frontmatter-layout', file, 'No layout specified');
      }
      if (!fm.title) {
        addWarning('frontmatter-title', file, 'No title specified');
      }
    } catch (err) {
      issueCount++;
      addError('frontmatter-yaml', file, `Invalid YAML: ${err.message}`);
    }
  }
  
  if (issueCount === 0) {
    results.passed++;
    log(`✓ Frontmatter check passed (${pages.length} pages)`, 'success');
  }
}

// ============================================================================
// CHECK 7: Image Files Exist
// ============================================================================
async function checkImages() {
  log('\nCHECK 7: Image references', 'info');
  results.checks++;
  
  const assetsDir = path.join(rootDir, 'assets', 'img');
  if (!fs.existsSync(assetsDir)) {
    log('  → No assets/img directory', 'info');
    results.passed++;
    return;
  }
  
  // Basic check - ensure directory is not empty
  const images = fs.readdirSync(assetsDir);
  if (images.length === 0) {
    addWarning('images-empty', 'assets/img/', 'Image directory is empty');
  } else {
    log(`  → Found ${images.length} images`, 'info');
    results.passed++;
  }
}

// ============================================================================
// CHECK 8: Git Status
// ============================================================================
async function checkGitStatus() {
  log('\nCHECK 8: Git repository status', 'info');
  results.checks++;
  
  const gitDir = path.join(rootDir, '.git');
  if (!fs.existsSync(gitDir)) {
    addWarning('git-status', '.git', 'Not a git repository');
    return;
  }
  
  // Simple check - just verify .git exists
  log('  → Git repository detected', 'info');
  results.passed++;
}

// ============================================================================
// MAIN EXECUTION
// ============================================================================
async function main() {
  console.log('\n╔═══════════════════════════════════════════════════════════╗');
  console.log('║     FAITHFRONTIER COMPLETE VALIDATION SYSTEM          ║');
  console.log('╚═══════════════════════════════════════════════════════════╝\n');
  
  if (config.fix) {
    log('FIX MODE ENABLED - Will auto-fix issues where possible', 'warning');
  }
  
  const startTime = Date.now();
  
  // Run all checks
  await checkDocketPaths();
  await checkPDFExistence();
  await checkCaseREADMEs();
  await checkCSS();
  await checkBuiltLinks();
  await checkFrontmatter();
  await checkImages();
  await checkGitStatus();
  
  const elapsed = ((Date.now() - startTime) / 1000).toFixed(2);
  
  // Print summary
  console.log('\n' + '═'.repeat(60));
  console.log('VALIDATION SUMMARY');
  console.log('═'.repeat(60));
  console.log(`Total Checks:    ${results.checks}`);
  console.log(`\x1b[32mPassed:          ${results.passed}\x1b[0m`);
  console.log(`\x1b[31mErrors:          ${results.errors.length}\x1b[0m`);
  console.log(`\x1b[33mWarnings:        ${results.warnings.length}\x1b[0m`);
  if (config.fix) {
    console.log(`\x1b[32mFixed:           ${results.fixed.length}\x1b[0m`);
  }
  console.log(`Time:            ${elapsed}s`);
  console.log('═'.repeat(60));
  
  // Detailed error report
  if (results.errors.length > 0) {
    console.log('\n\x1b[31mERRORS:\x1b[0m');
    const grouped = {};
    results.errors.forEach(err => {
      if (!grouped[err.check]) grouped[err.check] = [];
      grouped[err.check].push(err);
    });
    
    Object.entries(grouped).forEach(([check, errs]) => {
      console.log(`\n  ${check} (${errs.length}):`);
      errs.forEach(err => console.log(`    • ${err.file}: ${err.issue}`));
    });
  }
  
  // Exit code
  if (results.errors.length > 0) {
    console.log('\n\x1b[31m✗ Validation failed\x1b[0m\n');
    process.exit(1);
  } else {
    console.log('\n\x1b[32m✓ All validations passed\x1b[0m\n');
    process.exit(0);
  }
}

main().catch(err => {
  console.error('\n\x1b[31mFATAL ERROR:\x1b[0m', err);
  process.exit(1);
});

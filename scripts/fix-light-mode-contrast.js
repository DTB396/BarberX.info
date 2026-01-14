#!/usr/bin/env node
/**
 * Light Mode Contrast Fixer
 * Finds and fixes white/cream text on light backgrounds
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.join(__dirname, '..');

function* walkCSS(dir) {
  const files = fs.readdirSync(dir, { withFileTypes: true });
  for (const file of files) {
    const fullPath = path.join(dir, file.name);
    if (file.isDirectory() && !file.name.startsWith('.') && !file.name.startsWith('node_modules') && !file.name.startsWith('_site')) {
      yield* walkCSS(fullPath);
    } else if (file.isFile() && file.name.endsWith('.css')) {
      yield fullPath;
    }
  }
}

// Problematic patterns: light text that might appear on light backgrounds
const lightModeFixes = {
  // White text should be dark in light mode
  'color: white': 'color: var(--ink-900)',
  'color: #fff': 'color: var(--ink-900)',
  'color: #ffffff': 'color: var(--ink-900)',
  'color: rgba(255, 255, 255': 'color: var(--ink-900',
  
  // Cream text should be dark in light mode
  'color: var(--cream-50)': 'color: var(--ink-900)',
  'color: var(--cream-100)': 'color: var(--ink-700)',
  'color: var(--muted-300)': 'color: var(--ink-700)',
  
  // Ensure light mode class uses dark text
  '[data-theme="light"] .premium-hero__card-text {\n  color: var(--ink-700)': '[data-theme="light"] .premium-hero__card-text {\n  color: var(--ink-900)',
};

console.log('\n' + '='.repeat(90));
console.log('LIGHT MODE CONTRAST FIXER');
console.log('Fixing white/cream text on light backgrounds');
console.log('='.repeat(90) + '\n');

const cssDir = path.join(rootDir, 'assets', 'css');
const stats = {
  filesScanned: 0,
  filesFixed: 0,
  replacements: 0
};

for (const cssFile of walkCSS(cssDir)) {
  stats.filesScanned++;
  const relativePath = path.relative(rootDir, cssFile);
  
  let content = fs.readFileSync(cssFile, 'utf8');
  let modified = false;
  let fileReplacements = 0;
  
  // Check if this file has light mode styles
  const hasLightMode = content.includes('[data-theme="light"]') || 
                       content.includes('.light-mode') ||
                       content.includes('.ff-home');
  
  if (hasLightMode) {
    // Apply fixes
    for (const [find, replace] of Object.entries(lightModeFixes)) {
      if (content.includes(find)) {
        const count = (content.match(new RegExp(find.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'g')) || []).length;
        content = content.replaceAll(find, replace);
        modified = true;
        fileReplacements += count;
      }
    }
    
    if (modified) {
      fs.writeFileSync(cssFile, content, 'utf8');
      console.log(`✅ ${relativePath} - ${fileReplacements} fixes`);
      stats.filesFixed++;
      stats.replacements += fileReplacements;
    }
  }
}

console.log('\n' + '='.repeat(90));
console.log('📊 RESULTS');
console.log('='.repeat(90));
console.log(`Files Scanned:  ${stats.filesScanned}`);
console.log(`Files Fixed:    ${stats.filesFixed}`);
console.log(`Total Fixes:    ${stats.replacements}`);
console.log('='.repeat(90) + '\n');

if (stats.filesFixed > 0) {
  console.log('✅ Light mode contrast issues fixed!');
  console.log('\n📋 Next: Rebuild site and test');
} else {
  console.log('✅ No light mode contrast issues found!');
}

console.log('\n');

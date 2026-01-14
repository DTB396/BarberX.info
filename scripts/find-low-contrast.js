#!/usr/bin/env node
/**
 * Find Low Contrast Elements
 * Scans CSS for potentially problematic color combinations
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Problematic patterns to look for
const lowContrastPatterns = [
  { pattern: /color:\s*var\(--muted-400\)/gi, desc: 'muted-400 (low contrast on many backgrounds)' },
  { pattern: /color:\s*var\(--stone-200\)/gi, desc: 'stone-200 (very low contrast)' },
  { pattern: /color:\s*rgba\(168,\s*162,\s*158/gi, desc: 'muted-400 rgba (low contrast)' },
  { pattern: /color:\s*#a8a29e/gi, desc: 'muted-400 hex (low contrast)' },
  { pattern: /color:\s*rgba\(203,\s*213,\s*225/gi, desc: 'muted-300 rgba (may be low on light bg)' },
  { pattern: /opacity:\s*0\.[1-4]/gi, desc: 'Very low opacity (may cause readability issues)' },
];

const cssDir = path.join(__dirname, '..', 'assets', 'css');
const issues = [];

function scanFile(filePath) {
  const content = fs.readFileSync(filePath, 'utf8');
  const relativePath = path.relative(process.cwd(), filePath);
  
  for (const { pattern, desc } of lowContrastPatterns) {
    const matches = content.matchAll(pattern);
    for (const match of matches) {
      const lines = content.substring(0, match.index).split('\n');
      const lineNumber = lines.length;
      const line = content.split('\n')[lineNumber - 1];
      
      issues.push({
        file: relativePath,
        line: lineNumber,
        code: line.trim(),
        issue: desc
      });
    }
  }
}

function walkDir(dir) {
  const files = fs.readdirSync(dir);
  for (const file of files) {
    const fullPath = path.join(dir, file);
    const stat = fs.statSync(fullPath);
    
    if (stat.isDirectory()) {
      walkDir(fullPath);
    } else if (file.endsWith('.css')) {
      scanFile(fullPath);
    }
  }
}

console.log('\n' + '='.repeat(90));
console.log('LOW CONTRAST SCAN RESULTS');
console.log('='.repeat(90) + '\n');

walkDir(cssDir);

if (issues.length === 0) {
  console.log('✅ No obvious low-contrast patterns found!\n');
} else {
  console.log(`Found ${issues.length} potential contrast issues:\n`);
  
  for (const issue of issues) {
    console.log(`❌ ${issue.file}:${issue.line}`);
    console.log(`   Issue: ${issue.issue}`);
    console.log(`   Code: ${issue.code}`);
    console.log('');
  }
  
  console.log('\n' + '='.repeat(90));
  console.log('RECOMMENDATIONS:');
  console.log('='.repeat(90) + '\n');
  console.log('Light mode text colors:');
  console.log('  - Use var(--ink-900) for body text (17.21:1 contrast on white)');
  console.log('  - Use var(--ink-700) for muted text (11.70:1 contrast on white)');
  console.log('  - Use var(--emerald-700) for accents (7.92:1 contrast on white)');
  console.log('');
  console.log('Dark mode text colors:');
  console.log('  - Use var(--cream-50) for body text (18.59:1 contrast on navy950)');
  console.log('  - Use var(--muted-300) for muted text (13.09:1 contrast on navy950)');
  console.log('  - Use var(--emerald-300) for accents (12.75:1 contrast on navy950)');
  console.log('');
}

console.log('='.repeat(90) + '\n');
